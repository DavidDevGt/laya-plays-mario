"""Fine-tune Laya's typed-decisions checkpoint to imitate RuleTeacherV2 on Mario.

Design decisions (documented, not just asserted):
  - Initialize from the `typed-decisions` checkpoint (already fine-tuned once; closest available
    starting point), not from scratch and not from the plain English root checkpoint.
  - Freeze the ModernBERT-large encoder (395M params) and train only the small decision head
    (2-layer transformer + scorer + type embedding + act head, a few M params). Rationale: our
    dataset is small (~4k decisions from a handful of levels) -- full/LoRA fine-tuning of the full
    encoder risks catastrophic forgetting and overfitting at this scale. This also mirrors the
    reference blog's own recipe ("first train only the scoring layer, then LoRA"): we are doing
    step one and treating full unfreezing as a follow-up if head-only underfits.
  - Loss = negative `proper_reward` (log score + spherical score, the same strictly-proper scoring
    rule Laya's own RLCD training optimizes), backpropagated directly through the softmax rather
    than sampled via REINFORCE. This is valid and simpler than full RLCD specifically because we
    HAVE ground-truth target distributions (one-hot teacher labels) -- RLCD's sampling/exploration
    machinery exists for when you only have a scalar reward, not a known target; distillation with
    a known target should just differentiate through the proper scoring rule directly.
  - Validation is by held-out STAGE (not a random decision split): stages 4-1 and 7-1 are excluded
    from training entirely, so validation accuracy measures generalization to unseen terrain, not
    memorization of nearby frames in the same trajectory.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from rl_agent_api import RLAgent
from rl_common import build_model, collate_items, encode_record, proper_reward, seed_all
from mario.questions import QUESTIONS, MOVE_LABELS

BASE_CKPT = os.path.join(os.path.dirname(__file__), "..", "typed-decisions")
VAL_STAGES = {"SuperMarioBros-4-1-v0", "SuperMarioBros-7-1-v0"}


def load_records(jsonl_path):
    """Only the `jump` question is trained on. RuleTeacherV2's `move` label is a hardcoded
    constant ("right", always -- see teacher.py), so there is no real signal to learn there; a
    first fine-tune that included it produced a degenerate near-50/50 move head that latched onto
    `recent_actions` as a spurious shortcut and got stuck oscillating (see runs/logs/phase4_diagnosis.md).
    The controller hardcodes move="right" itself, matching what the teacher actually does.
    """
    jump_q_template = RLAgent._to_internal(QUESTIONS["jump"])
    train_recs, val_recs = [], []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            jump_q = dict(jump_q_template)
            jump_q["y"] = 1 if d["label_jump"] else 0
            rec = {"state": d["state"], "qs": [jump_q], "src": d["stage"]}
            (val_recs if d["stage"] in VAL_STAGES else train_recs).append(rec)
    return train_recs, val_recs


def build_items(recs, tok, cfg, rng, train: bool):
    items = []
    for rec in recs:
        items.extend(encode_record(rec, tok, cfg, rng, train))
    return items


def batch_iter(items, batch_size, rng):
    idx = list(range(len(items)))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [items[j] for j in idx[i:i + batch_size]]
        yield collate_items([chunk], pad_id=0)


@torch.no_grad()
def evaluate(model, items, device, dtype, pad_id, batch_size=128):
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct = {0: 0, 2: 0}  # qtype -> correct count (0=choice/move, 2=noul/jump)
    total = {0: 0, 2: 0}
    for b in batch_iter(items, batch_size, random.Random(0)):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        probs = torch.softmax(logits.float(), -1)
        target = b["target"].to(device)
        mask = b["marker_mask"].to(device)
        qtype = b["qtype"].to(device)
        loss = -proper_reward(probs, target, qtype, mask.float()).mean()
        total_loss += loss.item()
        n_batches += 1
        pred = probs.argmax(-1).cpu()
        label = b["label"]
        for qt in (0, 2):
            sel = (b["qtype"] == qt) & (label >= 0)
            if sel.any():
                correct[qt] += int((pred[sel] == label[sel]).sum())
                total[qt] += int(sel.sum())
    model.train()
    acc = {qt: (correct[qt] / total[qt] if total[qt] else float("nan")) for qt in (0, 2)}
    return total_loss / max(1, n_batches), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "teacher_rollouts_v1.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "mario_laya_v1"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    with open(os.path.join(BASE_CKPT, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(BASE_CKPT, "tokenizer"))

    model = build_model(cfg, encoder_dir=os.path.join(BASE_CKPT, "encoder"))
    model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    model.to(device)

    for p in model.encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%)")

    train_recs, val_recs = load_records(args.data)
    print(f"Records: {len(train_recs)} train, {len(val_recs)} val (held-out stages: {sorted(VAL_STAGES)})")

    rng = random.Random(args.seed)
    train_items = build_items(train_recs, tok, cfg, rng, train=True)
    val_items = build_items(val_recs, tok, cfg, None, train=False)
    print(f"Items (questions): {len(train_items)} train, {len(val_items)} val")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    model.train()

    print("\n--- Baseline (before any fine-tuning) ---")
    val_loss0, val_acc0 = evaluate(model, val_items, device, dtype, tok.pad_token_id)
    print(f"val_loss={val_loss0:.4f} move_acc={val_acc0[0]:.3f} jump_acc={val_acc0[2]:.3f}")

    history = [{"epoch": 0, "val_loss": val_loss0, "val_move_acc": val_acc0[0], "val_jump_acc": val_acc0[2]}]
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_items)
        epoch_loss, n_b = 0.0, 0
        for b in batch_iter(train_items, args.batch_size, rng):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
            probs = torch.softmax(logits.float(), -1)
            target = b["target"].to(device)
            mask = b["marker_mask"].to(device)
            qtype = b["qtype"].to(device)
            loss = -proper_reward(probs, target, qtype, mask.float()).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_b += 1
        val_loss, val_acc = evaluate(model, val_items, device, dtype, tok.pad_token_id)
        print(f"epoch {epoch:2d}  train_loss={epoch_loss / n_b:.4f}  val_loss={val_loss:.4f}  "
              f"move_acc={val_acc[0]:.3f}  jump_acc={val_acc[2]:.3f}  ({time.time() - t0:.0f}s elapsed)")
        history.append({"epoch": epoch, "train_loss": epoch_loss / n_b, "val_loss": val_loss,
                        "val_move_acc": val_acc[0], "val_jump_acc": val_acc[2]})

    os.makedirs(args.out, exist_ok=True)
    save_file(model.state_dict(), os.path.join(args.out, "model.safetensors"))
    shutil.copytree(os.path.join(BASE_CKPT, "encoder"), os.path.join(args.out, "encoder"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(BASE_CKPT, "tokenizer"), os.path.join(args.out, "tokenizer"), dirs_exist_ok=True)
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "training_log.json"), "w") as f:
        json.dump({"args": vars(args), "n_train_records": len(train_recs), "n_val_records": len(val_recs),
                   "val_stages": sorted(VAL_STAGES), "history": history}, f, indent=2)
    print(f"\nSaved checkpoint + training log to {args.out}")


if __name__ == "__main__":
    main()
