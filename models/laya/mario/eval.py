"""Rigorous policy evaluation by actually playing episodes -- per the reference blog's central
lesson, offline label accuracy does not predict this, so it must be measured directly.

Note on "consistency between episodes": this environment and every policy here (teacher, Laya at
argmax, random with a fixed seed) are deterministic. Re-running the identical policy on the
identical stage always produces byte-identical trajectories -- there is no meaningful run-to-run
variance to report there, and pretending otherwise would be dishonest. "Consistency" is instead
measured across the 8 different stages (real terrain/enemy diversity), reported as the standard
deviation of distance across stages alongside the mean.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("USE_TF", "0")

from rl_agent_api import RLAgent
from mario.controller import make_env, run_episode
from mario.state import MarioStateExtractor
from mario.teacher import RuleTeacherV1, RuleTeacherV2
from mario.questions import QUESTIONS

JUMP_ONLY_QUESTIONS = {"jump": QUESTIONS["jump"]}


def laya_decide_fn(agent, threshold, timing, hold_decisions: int = 0, stuck_threshold: int = 0):
    """move is hardcoded "right" (see finetune.py's load_records docstring for why the move head
    was dropped); only jump comes from the model. `timing` is a list this appends per-call
    wall-clock seconds to, isolating pure model inference latency from emulator step time.

    `hold_decisions`: Laya re-evaluates P(jump) independently every decision, so on multi-enemy
    gauntlets that need one long sustained jump (the ones that made RuleTeacherV2 need
    hold_decisions=7, see phase8_speedrun.md) its probability can dip below threshold mid-arc and
    cut the jump short. This is a controller-level fix, not a model change: once Laya commits to
    a jump, hold it for N more decisions before letting a new prediction override -- the same
    hysteresis mechanism the rule-based teacher already has built in.

    `stuck_threshold`: without this, Laya can get permanently stuck against a wall/pipe -- observed
    directly (see runs/logs/phase9_stuck_wall.md): P(jump) can hover just under threshold (e.g.
    0.39-0.43 against a 0.5 cutoff) indefinitely while velocity_x stays 0, because nothing forces
    a jump the way RuleTeacherV2's own stuck_threshold does. If on_ground and velocity_x==0 for
    this many consecutive decisions, force a jump regardless of P(jump) -- same fallback mechanism
    the rule-based teacher already has, ported to Laya's controller.
    """
    hold_remaining = [0]
    stuck_count = [0]

    def decide_fn(state):
        if hold_remaining[0] > 0:
            hold_remaining[0] -= 1
            return "right", True, "held"
        if state["on_ground"] and state["velocity_x"] == 0:
            stuck_count[0] += 1
        else:
            stuck_count[0] = 0
        if stuck_threshold > 0 and stuck_count[0] >= stuck_threshold:
            stuck_count[0] = 0
            if hold_decisions > 0:
                hold_remaining[0] = hold_decisions - 1
            return "right", True, "stuck-forced-jump"
        t0 = time.perf_counter()
        result = agent.system_one(state, JUMP_ONLY_QUESTIONS)
        timing.append(time.perf_counter() - t0)
        p = result["answers"]["jump"]["noul"]
        jump = p > threshold
        if jump and hold_decisions > 0:
            hold_remaining[0] = hold_decisions - 1
        return "right", jump, f"jump_p={p:.2f}"
    return decide_fn


def rule_decide_fn(teacher):
    def decide_fn(state):
        out = teacher.decide(state)
        return out[0], out[1], ""
    return decide_fn


def random_decide_fn(rng):
    def decide_fn(state):
        return rng.choice(["left", "right", "neutral"]), rng.random() < 0.3, ""
    return decide_fn


def run_one(policy, checkpoint, threshold, stage, ep, seed, action_repeat, max_steps, run, hold_decisions=0, stuck_threshold=0):
    env = make_env(stage)
    extractor = MarioStateExtractor()
    timing = []
    if policy == "laya":
        decide_fn = laya_decide_fn(checkpoint, threshold, timing, hold_decisions, stuck_threshold)  # checkpoint is the loaded RLAgent here
    elif policy == "rule_v1":
        decide_fn = rule_decide_fn(RuleTeacherV1())
    elif policy == "rule_v2":
        decide_fn = rule_decide_fn(RuleTeacherV2())
    else:
        decide_fn = random_decide_fn(random.Random(seed + ep))

    t0 = time.time()
    r = run_episode(env, decide_fn, extractor, action_repeat=action_repeat,
                    max_steps=max_steps, verbose=False, run=run)
    wall = time.time() - t0
    env.close()
    r.update({"stage": stage, "episode": ep})
    if timing:
        r["inference_ms_mean"] = round(1000 * sum(timing) / len(timing), 3)
        r["inference_ms_p95"] = round(1000 * sorted(timing)[int(0.95 * len(timing))], 3)
    r["actions_per_sec_wall"] = round(r["decisions"] / wall, 1) if wall > 0 else 0.0
    return r


def summarize(results):
    n = len(results)
    distances = [r["distance"] for r in results]
    completions = [r for r in results if r["flag_get"]]
    mean_d = sum(distances) / n
    var_d = sum((d - mean_d) ** 2 for d in distances) / n
    return {
        "n_episodes": n,
        "completion_rate": len(completions) / n,
        "avg_distance": round(mean_d, 1),
        "std_distance_across_stages": round(var_d ** 0.5, 1),
        "best_distance": max(distances),
        "deaths": sum(r["is_dead"] for r in results),
        "avg_time_used_all": round(sum(r["time_used"] for r in results) / n, 1),
        "avg_time_used_completions": round(sum(r["time_used"] for r in completions) / len(completions), 1) if completions else None,
        "best_time_used_completion": min((r["time_used"] for r in completions), default=None),
        "avg_frames": round(sum(r["frames"] for r in results) / n, 1),
        "avg_actions_per_sec_wall": round(sum(r["actions_per_sec_wall"] for r in results) / n, 1),
        "avg_inference_ms": round(sum(r.get("inference_ms_mean", 0) for r in results) / n, 3) if "inference_ms_mean" in results[0] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["laya", "rule_v1", "rule_v2", "random"])
    ap.add_argument("--checkpoint", default=None, help="required for --policy laya")
    ap.add_argument("--threshold", type=float, default=0.5, help="laya jump decision threshold")
    ap.add_argument("--stages", nargs="+", default=["SuperMarioBros-1-1-v0"])
    ap.add_argument("--episodes-per-stage", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--action-repeat", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", action="store_true", help="hold B (run); default walking")
    ap.add_argument("--hold-decisions", type=int, default=0, help="laya: hold a committed jump for N extra decisions (controller-level hysteresis, see laya_decide_fn)")
    ap.add_argument("--stuck-threshold", type=int, default=0, help="laya: force a jump if stuck against a wall this many consecutive decisions (see laya_decide_fn)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    agent = None
    if args.policy == "laya":
        assert args.checkpoint, "--checkpoint required for --policy laya"
        agent = RLAgent(args.checkpoint)
        print(f"Loaded {args.checkpoint} on {agent.device}, threshold={args.threshold}, run={args.run}, hold_decisions={args.hold_decisions}, stuck_threshold={args.stuck_threshold}")

    results = []
    for stage in args.stages:
        for ep in range(args.episodes_per_stage):
            r = run_one(args.policy, agent if agent else None, args.threshold, stage, ep,
                       args.seed, args.action_repeat, args.max_steps, args.run, args.hold_decisions, args.stuck_threshold)
            results.append(r)
            extra = f" inf={r.get('inference_ms_mean', '-')}ms" if "inference_ms_mean" in r else ""
            print(f"[{args.policy}] {stage} ep{ep}: distance={r['distance']:4d} flag={r['flag_get']} "
                  f"dead={r['is_dead']} time_used={r['time_used']} frames={r['frames']} "
                  f"aps={r['actions_per_sec_wall']}{extra}")

    summary = summarize(results)
    summary.update({"policy": args.policy, "checkpoint": args.checkpoint, "threshold": args.threshold,
                    "run": args.run, "results": results})
    print(f"\n--- Summary: {args.policy} ---")
    for k, v in summary.items():
        if k != "results":
            print(f"  {k}: {v}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
