"""Laya as a game controller for Super Mario Bros.

Perception (emulator info dict) -> structured state -> Laya (decision) -> controller -> action.
This is a baseline: Laya's typed-decisions checkpoint has never seen Mario states, so treat
results as a starting point to beat with fine-tuning, not a demo of a trained agent.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("USE_TF", "0")

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from rl_agent_api import RLAgent

MODEL_DIR = os.path.join(os.path.dirname(__file__), "typed-decisions")

# SIMPLE_MOVEMENT = [NOOP, right, right+A, right+B, right+A+B, A, left]
NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A_ONLY, LEFT = range(7)

QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "Mario is playing a 2D side-scrolling platformer. Given the current game state, "
                        "which horizontal direction should Mario move to make progress and avoid danger?",
        "criteria": {
            "left": "move left, away from the goal",
            "right": "move right, towards the goal",
            "neutral": "stand still, do not move horizontally",
        },
    },
    "jump": {
        "type": "noul",
        "instructions": "Given the current game state, should Mario jump right now (to clear a gap, "
                        "land on an enemy, or avoid an obstacle)?",
    },
}


class MarioStateExtractor:
    """Turns the emulator's info dict (+ a short history) into a structured state for Laya."""

    def __init__(self, history_len: int = 5):
        self.history_len = history_len
        self.prev_x = None
        self.prev_y = None
        self.action_history = []

    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self.action_history = []

    def record_action(self, action_name: str):
        self.action_history.append(action_name)
        self.action_history = self.action_history[-self.history_len:]

    def build(self, info: dict) -> dict:
        x, y = info["x_pos"], info["y_pixel"]
        vx = 0 if self.prev_x is None else x - self.prev_x
        vy = 0 if self.prev_y is None else y - self.prev_y
        self.prev_x, self.prev_y = x, y
        enemies_on_screen = sum(1 for e in info.get("enemy_types", ()) if e != 0)
        return {
            "mario_x": x,
            "mario_y": y,
            "velocity_x": vx,
            "velocity_y": vy,
            "on_ground": vy == 0,
            "enemies_on_screen": enemies_on_screen,
            "time_remaining": info["time"],
            "world": info["world"],
            "stage": info["stage"],
            "recent_actions": self.action_history[-3:],
        }


def action_from_answers(answers: dict) -> tuple[int, str]:
    move = answers["move"]["choice"]
    jump = answers["jump"]["noul"] > 0.5
    return action_from_move_jump(move, jump)


def action_from_move_jump(move: str, jump: bool) -> tuple[int, str]:
    if move == "right":
        return (RIGHT_A if jump else RIGHT), ("right+jump" if jump else "right")
    if move == "left":
        return LEFT, "left"
    return (A_ONLY if jump else NOOP), ("jump" if jump else "noop")


class RuleTeacher:
    """Hand-fitted heuristic: hold right, jump reactively near enemies and periodically to clear pits/pipes.

    No tile-map reading (would need SMB RAM address tables) -- this only uses the same info-dict fields
    the extractor already exposes, on purpose, so it is a fair baseline for what Laya's state actually sees.
    Jump height in SMB depends on how long A is held, so once triggered we hold it for `hold_decisions`
    consecutive decisions instead of releasing immediately.
    """

    def __init__(self, jump_period: int = 38, hold_decisions: int = 3):
        self.jump_period = jump_period
        self.hold_decisions = hold_decisions
        self.frames_since_jump = 0
        self.hold_remaining = 0

    def reset(self):
        self.frames_since_jump = 0
        self.hold_remaining = 0

    def decide(self, state: dict) -> tuple[str, bool]:
        self.frames_since_jump += 1
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            return "right", True
        jump = False
        if state["on_ground"] and state["enemies_on_screen"] > 0:
            jump = True
        elif state["on_ground"] and self.frames_since_jump >= self.jump_period:
            jump = True
        if jump:
            self.frames_since_jump = 0
            self.hold_remaining = self.hold_decisions - 1
        return "right", jump


def run_episode(env, decide_fn, extractor: MarioStateExtractor, action_repeat: int, max_steps: int, verbose: bool):
    """decide_fn(state) -> (move: str, jump: bool, log_extra: str)"""
    obs, info = env.reset()
    extractor.reset()
    total_frames, decisions, t0 = 0, 0, time.time()
    while total_frames < max_steps:
        state = extractor.build(info)
        move, jump, log_extra = decide_fn(state)
        action_id, action_name = action_from_move_jump(move, jump)
        extractor.record_action(action_name)
        decisions += 1
        if verbose:
            print(f"  frame={total_frames:5d} state={state} -> {action_name} {log_extra}")
        for _ in range(action_repeat):
            obs, reward, terminated, truncated, info = env.step(action_id)
            total_frames += 1
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    elapsed = time.time() - t0
    return {
        "flag_get": info["flag_get"],
        "is_dead": info["is_dead"] or info["is_game_over"] or info.get("death", False),
        "distance": info["x_pos"],
        "time_left": info["time"],
        "frames": total_frames,
        "decisions": decisions,
        "wall_seconds": round(elapsed, 1),
        "decisions_per_sec": round(decisions / elapsed, 1) if elapsed > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["laya", "rule", "random"], default="laya")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--action-repeat", type=int, default=4, help="frames to hold each decision")
    ap.add_argument("--max-steps", type=int, default=2000, help="max emulator frames per episode")
    ap.add_argument("--level", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    env = gym_super_mario_bros.make(args.level)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    extractor = MarioStateExtractor()

    if args.policy == "laya":
        print("Loading Laya (typed-decisions checkpoint)...")
        agent = RLAgent(MODEL_DIR)
        print(f"  device: {agent.device}")

        def decide_fn(state):
            result = agent.system_one(state, QUESTIONS)
            move = result["answers"]["move"]["choice"]
            jump = result["answers"]["jump"]["noul"] > 0.5
            log = f"(move={result['answers']['move']['probabilities']}, jump={result['answers']['jump']['noul']:.2f})"
            return move, jump, log

    elif args.policy == "rule":
        teacher = RuleTeacher()
        orig_reset = extractor.reset

        def reset_both():
            orig_reset()
            teacher.reset()

        extractor.reset = reset_both

        def decide_fn(state):
            move, jump = teacher.decide(state)
            return move, jump, ""

    else:  # random
        import random as _random

        def decide_fn(state):
            move = _random.choice(["left", "right", "neutral"])
            jump = _random.random() < 0.3
            return move, jump, ""

    results = []
    for ep in range(1, args.episodes + 1):
        print(f"\nEpisode {ep}/{args.episodes}")
        r = run_episode(env, decide_fn, extractor, args.action_repeat, args.max_steps, args.verbose)
        results.append(r)
        outcome = "FLAG" if r["flag_get"] else ("DEAD" if r["is_dead"] else "TIMEOUT")
        print(f"  {outcome} | distance={r['distance']} | frames={r['frames']} | "
              f"decisions={r['decisions']} ({r['decisions_per_sec']}/s) | wall={r['wall_seconds']}s")

    env.close()

    print("\n--- Summary ---")
    n = len(results)
    print(f"Episodes: {n}")
    print(f"Flags reached: {sum(r['flag_get'] for r in results)}/{n}")
    print(f"Avg distance: {sum(r['distance'] for r in results) / n:.0f}")
    print(f"Avg decisions/sec: {sum(r['decisions_per_sec'] for r in results) / n:.1f}")


if __name__ == "__main__":
    main()
