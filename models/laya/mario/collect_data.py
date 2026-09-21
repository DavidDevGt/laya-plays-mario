"""Collect (state, teacher_action) imitation data from RuleTeacherV2.

Design notes (see runs/logs/phase1_teacher.md for why this matters):
  - The emulator is fully deterministic: replaying the same policy on the same stage gives the
    exact same trajectory every time. A single rollout per stage would produce a dataset with
    near-zero state diversity, which is a bad basis for a supervised policy (it would only ever
    see one path through the level).
  - We fix this two ways: (1) collect across multiple stages (different terrain/enemy layouts),
    and (2) inject exploration -- with probability `epsilon` we EXECUTE a random action instead of
    the teacher's, so the trajectory visits off-policy states, but we always LABEL the current
    state with what the teacher would have done. This is the core idea behind DAgger, applied
    here at data-collection time using the hand-written teacher as the oracle (a real DAgger round
    against the *learned* policy comes later, once there is a learned policy to correct).
  - Restricted to World-X-1 stages (standard overworld run/jump stages): the teacher's heuristics
    (ground-tile gaps, walking enemies) don't model underwater or castle/lava physics, and using
    stages it wasn't designed for would inject mislabeled-by-design data.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("USE_TF", "0")

from mario.controller import make_env, run_episode
from mario.state import MarioStateExtractor
from mario.teacher import RuleTeacherV2

DEFAULT_STAGES = [f"SuperMarioBros-{w}-1-v0" for w in range(1, 9)]


def collect(stages, episodes_per_stage: int, epsilon: float, seed: int, max_steps: int,
           action_repeat: int, out_path: str, run: bool = False):
    rng = random.Random(seed)
    n_written = 0
    stage_stats = []
    with open(out_path, "w") as f:
        for stage in stages:
            for ep in range(episodes_per_stage):
                env = make_env(stage)
                extractor = MarioStateExtractor()
                teacher = RuleTeacherV2()

                def decide_fn(state):
                    _t_out = teacher.decide(state)
                    t_move, t_jump = _t_out[0], _t_out[1]
                    if rng.random() < epsilon:
                        exec_move = rng.choice(["left", "right", "neutral"])
                        exec_jump = rng.random() < 0.3
                    else:
                        exec_move, exec_jump = t_move, t_jump
                    record = {"stage": stage, "episode": ep, "state": state,
                              "label_move": t_move, "label_jump": t_jump}
                    f.write(json.dumps(record) + "\n")
                    nonlocal_counter[0] += 1
                    return exec_move, exec_jump

                nonlocal_counter = [0]
                r = run_episode(env, decide_fn, extractor, action_repeat=action_repeat,
                                max_steps=max_steps, verbose=False, run=run)
                env.close()
                n_written += nonlocal_counter[0]
                stage_stats.append({"stage": stage, "episode": ep, **r, "decisions_written": nonlocal_counter[0]})
                print(f"  {stage} ep{ep}: distance={r['distance']} flag={r['flag_get']} "
                      f"dead={r['is_dead']} decisions={nonlocal_counter[0]}")
    return n_written, stage_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=DEFAULT_STAGES)
    ap.add_argument("--episodes-per-stage", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--action-repeat", type=int, default=4)
    ap.add_argument("--run", action="store_true", help="hold B (run); default is walking, the only mode teacher_v2 is verified to clear a level with")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "teacher_rollouts.jsonl"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Collecting: {len(args.stages)} stages x {args.episodes_per_stage} episodes, "
          f"epsilon={args.epsilon}, seed={args.seed}, run={args.run}")
    n, stats = collect(args.stages, args.episodes_per_stage, args.epsilon, args.seed,
                       args.max_steps, args.action_repeat, args.out, run=args.run)

    meta_path = args.out.replace(".jsonl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"args": vars(args), "n_decisions": n, "episodes": stats}, f, indent=2)
    print(f"\nWrote {n} decisions to {args.out}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
