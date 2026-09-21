# Phase 4/5: Fine-tuning and evaluation -- results log

## Checkpoints trained

| Name | Data | Questions trained | Outcome |
|---|---|---|---|
| `mario_laya_v1` | teacher_rollouts_v1.jsonl (buggy gap detector) | move + jump | **Failed**: move label is a hardcoded constant ("right", always -- RuleTeacherV2 never varies it), so the move head had no real signal, converged to a near-50/50 tie at the spawn state, and a single wrong argmax tie-break (`left`) self-reinforced via the `recent_actions` feature into a full stuck-at-wall failure (distance=0 on the very stage it trained on). Diagnosed by hardcoding move="right" at eval time and re-testing: distance jumped 0 -> 704, confirming the move head was actively harmful. |
| `mario_laya_v2_jumponly` | teacher_rollouts_v1.jsonl (still buggy) | jump only | Real gameplay distance 296 (1-1), 322 (4-1 held-out), 290 (7-1 held-out, survived full 4000 frames without dying). Offline val jump_acc reached 0.86-0.90. Root cause of the ~90% "gap_ahead" base rate in this data: a tile-reading vertical-offset bug (see phase1_correction.md) -- the model partially learned an over-eager "jump often" policy inherited from noisy labels. |
| **`mario_laya_v3_clean`** | teacher_rollouts_v2.jsonl (fixed gap detector) | jump only | Val jump_acc 0.806 (baseline) -> **0.925** (epoch 10) on held-out stages 4-1/7-1, monotonic improvement, not clearly plateaued. Real gameplay: distance 296 (1-1), 322 (4-1), 290 (7-1) -- **numerically identical to v2** despite different training data (confirmed via checkpoint md5 that these are genuinely different weight files, not a loading bug). |

## Diagnosis: why gameplay distance plateaus at ~296-322 regardless of these fixes

Swept the jump decision threshold (0.5, 0.35, 0.25, 0.15) against `mario_laya_v3_clean` on World 1-1:
death occurs at **exactly x=296 in every case**, despite completely different action sequences
(at threshold 0.5 the model barely jumps near the fatal enemy encounter, at 0.15 it jumps
continuously the whole approach via the same recent_actions feedback loop seen in the v1 failure).
Both extremes die at the identical spot.

This means the bottleneck is not simply "jump more" or "jump less" -- it's **timing precision**.
RuleTeacherV2 jumps at an exact, deterministic pixel threshold (`nearest_enemy_dx <= 28`) with a
sharp on/off trigger. Laya's learned policy instead outputs a *smooth, gradually-varying*
probability that correlates with enemy proximity (visible in the verbose trace: p climbs from
~0.15 at dx=140 to ~0.5 at dx=17) but never sharply commits at the one frame-window that would
clear this specific enemy without collision. A soft, smoothly-varying probability function is a
poor fit for a hard-threshold reactive rule when the collision geometry only tolerates a narrow
window -- this is a real, interpretable limitation, not a training bug.

**This is consistent with (and sharpens) the reference blog's central finding**: high offline
accuracy (92.5%) does not guarantee correct behavior at the specific frames that matter, because
platforming is a sequential-decision problem where one mistimed frame is fatal, and aggregate
accuracy doesn't weight those critical frames any higher than easy, low-stakes ones.

## What would plausibly fix this (not yet attempted, next steps)

1. **More/targeted data around near-miss encounters** -- oversample states with small
  `nearest_enemy_dx` (the decisive, safety-critical region) rather than uniform episode sampling.
2. **DAgger round** (Phase 7): run `mario_laya_v3_clean` itself, collect the states where it
  diverges from RuleTeacherV2, add those corrections, retrain. This directly targets the
  distribution-shift gap the blog highlights, and specifically targets the states this checkpoint
  gets wrong (like x=296) rather than the teacher's original trajectory.
3. **Unfreeze the encoder** (currently frozen; only ~6.3% of params are trainable) -- the head-only
  budget may not have enough capacity to sharpen a near-threshold decision boundary.
4. **More epochs** -- val accuracy was still climbing at epoch 10, not clearly converged.

None of these were run yet due to time/compute budgeting for this pass; recommended as the next
concrete increment (DAgger, option 2, is the most targeted and cheapest to try first).
