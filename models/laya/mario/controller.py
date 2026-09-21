"""Emulator plumbing: action mapping and episode rollout, shared by every policy (teacher, Laya, random)."""
import time

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from .state import MarioStateExtractor

NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A_ONLY, LEFT = range(7)


def make_env(level: str = "SuperMarioBros-1-1-v0"):
    env = gym_super_mario_bros.make(level)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def action_from_move_jump(move: str, jump: bool, run: bool = True) -> tuple[int, str]:
    """`run` holds the B button: faster top speed and longer/higher jump arcs in SMB1 physics."""
    if move == "right":
        if jump:
            return (RIGHT_A_B if run else RIGHT_A), ("right+run+jump" if run else "right+jump")
        return (RIGHT_B if run else RIGHT), ("right+run" if run else "right")
    if move == "left":
        return LEFT, "left"
    return (A_ONLY if jump else NOOP), ("jump" if jump else "noop")


def run_episode(env, decide_fn, extractor: MarioStateExtractor, action_repeat: int = 4,
                max_steps: int = 3500, verbose: bool = False, on_decision=None, run: bool = False):
    """decide_fn(state) -> (move, jump[, log_extra]).
    on_decision(state, move, jump, info), if given, is called after every decision (for data logging).
    """
    obs, info = env.reset()
    extractor.reset()
    ram = env.unwrapped.ram
    total_frames, decisions, t0 = 0, 0, time.time()
    while total_frames < max_steps:
        state = extractor.build(ram, info)
        out = decide_fn(state)
        move, jump = out[0], out[1]
        log_extra = out[2] if len(out) > 2 else ""
        use_run = out[3] if len(out) > 3 else run
        action_id, action_name = action_from_move_jump(move, jump, use_run)
        extractor.record_action(action_name)
        decisions += 1
        if on_decision is not None:
            on_decision(state, move, jump, info)
        if verbose:
            print(f"  frame={total_frames:5d} x={state['mario_x']:4d} state={state} -> {action_name} {log_extra}")
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
        "time_used": 400 - info["time"],  # SMB1's in-game clock counts down from 400
        "frames": total_frames,
        "decisions": decisions,
        "wall_seconds": round(elapsed, 1),
        "decisions_per_sec": round(decisions / elapsed, 1) if elapsed > 0 else 0.0,
    }
