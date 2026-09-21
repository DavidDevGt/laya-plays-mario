"""DAgger round: let the learned Laya policy drive, label every state it visits with what
RuleTeacherV2 would have done there (a "shadow" teacher instance that watches the same state
stream and advances its own internal hold/stuck counters accordingly), and write those pairs out.

This directly targets the distribution-shift gap identified in phase4_5_finetune_eval.md: the
original supervised dataset only contains states the TEACHER's own trajectory visited, which is
not the same as the states the LEARNER ends up in once its own (imperfect) policy is driving.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("USE_TF", "0")

from rl_agent_api import RLAgent
from mario.controller import make_env, run_episode
from mario.state import MarioStateExtractor
from mario.teacher import RuleTeacherV2
from mario.eval import JUMP_ONLY_QUESTIONS

DEFAULT_STAGES = [f"SuperMarioBros-{w}-1-v0" for w in range(1, 9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "runs", "mario_laya_v3_clean"))
    ap.add_argument("--stages", nargs="+", default=DEFAULT_STAGES)
    ap.add_argument("--episodes-per-stage", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.1, help="exploration on top of the learner's own policy")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--action-repeat", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "dagger_rollouts_v1.jsonl"))
    args = ap.parse_args()

    agent = RLAgent(args.checkpoint)
    print(f"Loaded learner {args.checkpoint} on {agent.device}")
    rng = random.Random(args.seed)

    n_written = 0
    episode_stats = []
    with open(args.out, "w") as f:
        for stage in args.stages:
            for ep in range(args.episodes_per_stage):
                env = make_env(stage)
                extractor = MarioStateExtractor()
                shadow_teacher = RuleTeacherV2()  # advances its own hold/stuck state over the LEARNER's trajectory

                def on_decision(state, move, jump, info, f=f, shadow_teacher=shadow_teacher, stage=stage, ep=ep):
                    _teach_out = shadow_teacher.decide(state)
                    label_move, label_jump = _teach_out[0], _teach_out[1]
                    f.write(json.dumps({"stage": stage, "episode": ep, "state": state,
                                        "label_move": label_move, "label_jump": label_jump,
                                        "source": "dagger"}) + "\n")

                def decide_fn(state, agent=agent, rng=rng, eps=args.epsilon):
                    result = agent.system_one(state, JUMP_ONLY_QUESTIONS)
                    p = result["answers"]["jump"]["noul"]
                    jump = p > 0.5
                    if rng.random() < eps:
                        jump = rng.random() < 0.3
                    return "right", jump, f"p={p:.2f}"

                r = run_episode(env, decide_fn, extractor, action_repeat=args.action_repeat,
                                max_steps=args.max_steps, verbose=False, on_decision=on_decision, run=False)
                env.close()
                n_written += r["decisions"]
                episode_stats.append({"stage": stage, "episode": ep, **r})
                print(f"  {stage} ep{ep}: distance={r['distance']} flag={r['flag_get']} "
                      f"dead={r['is_dead']} decisions={r['decisions']}")

    meta_path = args.out.replace(".jsonl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"args": vars(args), "n_decisions": n_written, "episodes": episode_stats}, f, indent=2)
    avg_dist = sum(e["distance"] for e in episode_stats) / len(episode_stats)
    print(f"\nWrote {n_written} DAgger decisions to {args.out}")
    print(f"Learner's own avg distance during collection: {avg_dist:.0f}")


if __name__ == "__main__":
    main()
