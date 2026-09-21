"""Run ONE real episode with the final calibrated Laya controller (checkpoint mario_laya_v6_dagger,
threshold=0.5, hold_decisions=4, stuck_threshold=8 -- see FINAL_REPORT.md) and record two things
that stay frame-accurately in sync because they come from the exact same loop:

  1. episode.mp4    -- the raw emulator frames, 60fps, via ffmpeg (same technique as record.py).
  2. episode.jsonl  -- one structured event per DECISION (every action_repeat=4 frames), with
                       every field the demo UI displays: game state, the model's real P(jump)
                       (or why there wasn't a fresh model call this decision -- held/stuck-fallback
                       are real controller behavior, not model output, and are labeled as such),
                       the selected action, which buttons are actually held, and timing.

No values are invented. Where the real pipeline does not produce a signal (e.g. no per-direction
move probabilities, since move is hardcoded; no run-button probability, since the final config
does not hold run) the field is simply absent or explicitly null -- the UI is responsible for
labeling those honestly, not for fabricating a plausible-looking number.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ.setdefault("USE_TF", "0")

from rl_agent_api import RLAgent
from mario.controller import make_env, action_from_move_jump, NOOP
from mario.state import MarioStateExtractor
from mario.questions import QUESTIONS

JUMP_ONLY_QUESTIONS = {"jump": QUESTIONS["jump"]}
FRAME_W, FRAME_H = 256, 240
FPS = 60

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "mario_laya_v6_dagger")
DEFAULT_THRESHOLD = 0.5
DEFAULT_HOLD = 4
DEFAULT_STUCK = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--hold-decisions", type=int, default=DEFAULT_HOLD)
    ap.add_argument("--stuck-threshold", type=int, default=DEFAULT_STUCK)
    ap.add_argument("--stage", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--action-repeat", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--tail-frames", type=int, default=150, help="extra frames after flag_get to capture the flagpole/castle animation")
    ap.add_argument("--out-dir", default=os.path.dirname(__file__))
    args = ap.parse_args()

    video_path = os.path.join(args.out_dir, "episode.mp4")
    jsonl_path = os.path.join(args.out_dir, "episode.jsonl")
    meta_path = os.path.join(args.out_dir, "episode_meta.json")

    print(f"Loading {args.checkpoint} ...")
    agent = RLAgent(args.checkpoint)
    print(f"  device={agent.device}  threshold={args.threshold}  hold_decisions={args.hold_decisions}  stuck_threshold={args.stuck_threshold}")

    env = make_env(args.stage)
    extractor = MarioStateExtractor()
    obs, info = env.reset()
    ram = env.unwrapped.ram

    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{FRAME_W}x{FRAME_H}", "-framerate", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", video_path,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ffmpeg.stdin.write(obs.tobytes())

    events = []
    hold_remaining = 0
    stuck_count = 0
    last_real_p = None
    last_real_decision_index = None
    total_frames, decision_index = 0, 0
    t_wall_start = time.time()

    while total_frames < args.max_steps:
        state = extractor.build(ram, info)
        frame_start = total_frames

        # --- replicate laya_decide_fn's exact logic (eval.py), but log every branch ---
        model_latency_ms = None
        if hold_remaining > 0:
            hold_remaining -= 1
            jump = True
            decision_source = "held"
            p = None
        else:
            if state["on_ground"] and state["velocity_x"] == 0:
                stuck_count += 1
            else:
                stuck_count = 0
            if args.stuck_threshold > 0 and stuck_count >= args.stuck_threshold:
                stuck_count = 0
                jump = True
                decision_source = "stuck_fallback"
                p = None
                if args.hold_decisions > 0:
                    hold_remaining = args.hold_decisions - 1
            else:
                t0 = time.perf_counter()
                result = agent.system_one(state, JUMP_ONLY_QUESTIONS)
                model_latency_ms = (time.perf_counter() - t0) * 1000
                p = result["answers"]["jump"]["noul"]
                jump = p > args.threshold
                decision_source = "model"
                last_real_p, last_real_decision_index = p, decision_index
                if jump and args.hold_decisions > 0:
                    hold_remaining = args.hold_decisions - 1
        move = "right"
        # -----------------------------------------------------------------------------

        action_id, action_name = action_from_move_jump(move, jump, False)  # run=False: final config
        extractor.record_action(action_name)
        buttons = ["RIGHT"] + (["A"] if jump else [])

        events.append({
            "decision_index": decision_index,
            "frame_start": frame_start,
            "timestamp_s": round(frame_start / FPS, 4),
            "mario_x": state["mario_x"], "mario_y": state["mario_y"],
            "velocity_x": state["velocity_x"], "velocity_y": state["velocity_y"],
            "on_ground": state["on_ground"],
            "nearest_enemy_dx": state["nearest_enemy_dx"], "nearest_enemy_dy": state["nearest_enemy_dy"],
            "nearest_enemy_type": state["nearest_enemy_type"],
            "num_enemies_onscreen": state["num_enemies_onscreen"],
            "gap_ahead": state["gap_ahead"],
            "game_time_used": 400 - info["time"],
            "jump_probability": round(p, 4) if p is not None else None,
            "decision_source": decision_source,  # "model" | "held" | "stuck_fallback"
            "last_real_jump_probability": round(last_real_p, 4) if last_real_p is not None else None,
            "last_real_decision_index": last_real_decision_index,
            "selected_move": move,
            "selected_jump": jump,
            "buttons": buttons,
            "hold_frames_remaining": hold_remaining,
            "model_latency_ms": round(model_latency_ms, 3) if model_latency_ms is not None else None,
        })

        terminated = truncated = False
        for _ in range(args.action_repeat):
            obs, reward, terminated, truncated, info = env.step(action_id)
            ffmpeg.stdin.write(obs.tobytes())
            total_frames += 1
            if terminated or truncated:
                break
        decision_index += 1
        if terminated or truncated:
            break

    outcome = "FLAG" if info["flag_get"] else ("DEAD" if (info["is_dead"] or info.get("death")) else "TIMEOUT")
    final_distance, final_time_used = info["x_pos"], 400 - info["time"]

    if info["flag_get"] and args.tail_frames > 0:
        for _ in range(args.tail_frames):
            env.unwrapped.done = False
            obs, reward, terminated, truncated, info = env.step(NOOP)
            ffmpeg.stdin.write(obs.tobytes())
            total_frames += 1

    ffmpeg.stdin.close()
    ffmpeg.wait()
    env.close()
    wall_elapsed = time.time() - t_wall_start

    with open(jsonl_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    meta = {
        "checkpoint": args.checkpoint, "threshold": args.threshold, "hold_decisions": args.hold_decisions,
        "stuck_threshold": args.stuck_threshold, "stage": args.stage, "action_repeat": args.action_repeat,
        "outcome": outcome, "final_distance": final_distance, "final_time_used": final_time_used,
        "total_frames": total_frames, "total_decisions": len(events), "fps": FPS,
        "video_duration_s": round(total_frames / FPS, 2), "wall_seconds_to_record": round(wall_elapsed, 1),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{outcome} at distance={final_distance}, time_used={final_time_used}, decisions={len(events)}")
    print(f"Video:  {video_path}")
    print(f"Events: {jsonl_path}")
    print(f"Meta:   {meta_path}")


if __name__ == "__main__":
    main()
