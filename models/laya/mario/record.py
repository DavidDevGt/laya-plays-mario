"""Record an episode as an MP4, piping raw frames straight into ffmpeg (no extra image deps)."""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("USE_TF", "0")

from rl_agent_api import RLAgent
from mario.controller import make_env, action_from_move_jump, NOOP
from mario.state import MarioStateExtractor
from mario.teacher import RuleTeacherV1, RuleTeacherV2
from mario.eval import JUMP_ONLY_QUESTIONS

FRAME_W, FRAME_H = 256, 240
FPS = 60


def make_decide_fn(policy, checkpoint, threshold, run, hold_decisions=0, stuck_threshold=0):
    if policy == "laya":
        agent = RLAgent(checkpoint)
        print(f"Loaded {checkpoint} on {agent.device}, threshold={threshold}, run={run}, "
              f"hold_decisions={hold_decisions}, stuck_threshold={stuck_threshold}")
        hold_remaining = [0]
        stuck_count = [0]
        def decide_fn(state):
            if hold_remaining[0] > 0:
                hold_remaining[0] -= 1
                return "right", True, "held", run
            if state["on_ground"] and state["velocity_x"] == 0:
                stuck_count[0] += 1
            else:
                stuck_count[0] = 0
            if stuck_threshold > 0 and stuck_count[0] >= stuck_threshold:
                stuck_count[0] = 0
                if hold_decisions > 0:
                    hold_remaining[0] = hold_decisions - 1
                return "right", True, "stuck-forced-jump", run
            result = agent.system_one(state, JUMP_ONLY_QUESTIONS)
            p = result["answers"]["jump"]["noul"]
            jump = p > threshold
            if jump and hold_decisions > 0:
                hold_remaining[0] = hold_decisions - 1
            return "right", jump, f"P(jump)={p:.2f}", run
        return decide_fn
    teacher = RuleTeacherV2() if policy == "rule_v2" else RuleTeacherV1()
    def decide_fn(state):
        out = teacher.decide(state)
        move, jump = out[0], out[1]
        # `run` is explicit, not the teacher's own hybrid suggestion: walking (run=False) is the
        # only configuration verified to clear World 1-1 (see runs/logs/phase8_speedrun.md) --
        # running breaks the gap/enemy timing this teacher was tuned against.
        return move, jump, "", run
    return decide_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="laya", choices=["laya", "rule_v1", "rule_v2"])
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "runs", "mario_laya_v4_dagger"))
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--stage", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--action-repeat", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--scale", type=int, default=3, help="output video is scale*256 x scale*240")
    ap.add_argument("--run", action="store_true", help="hold B (run); default walking")
    ap.add_argument("--hold-decisions", type=int, default=0)
    ap.add_argument("--stuck-threshold", type=int, default=0)
    ap.add_argument("--tail-frames", type=int, default=150,
                    help="extra frames to record after flag_get (flagpole slide + castle walk-in); "
                         "the emulator sets terminated=True the instant flag_get flips, before that "
                         "animation plays, so this forces a few more steps past it (see NOOP; nes_py "
                         "refuses .step() once done, so we reset env.unwrapped.done each step)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "videos", "laya_play.mp4"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    decide_fn = make_decide_fn(args.policy, args.checkpoint, args.threshold, args.run, args.hold_decisions, args.stuck_threshold)

    env = make_env(args.stage)
    extractor = MarioStateExtractor()
    obs, info = env.reset()
    ram = env.unwrapped.ram

    out_w, out_h = FRAME_W * args.scale, FRAME_H * args.scale
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{FRAME_W}x{FRAME_H}", "-framerate", str(FPS), "-i", "-",
        "-vf", f"scale={out_w}:{out_h}:flags=neighbor",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_frames, decisions = 0, 0
    ffmpeg.stdin.write(obs.tobytes())
    while total_frames < args.max_steps:
        state = extractor.build(ram, info)
        out = decide_fn(state)
        move, jump = out[0], out[1]
        use_run = out[3] if len(out) > 3 else True
        action_id, action_name = action_from_move_jump(move, jump, use_run)
        extractor.record_action(action_name)
        decisions += 1
        for _ in range(args.action_repeat):
            obs, reward, terminated, truncated, info = env.step(action_id)
            ffmpeg.stdin.write(obs.tobytes())
            total_frames += 1
            if terminated or truncated:
                break
        if terminated or truncated:
            break

    outcome = "FLAG" if info["flag_get"] else ("DEAD" if (info["is_dead"] or info.get("death")) else "TIMEOUT")
    final_distance = info["x_pos"]

    if info["flag_get"] and args.tail_frames > 0:
        for _ in range(args.tail_frames):
            env.unwrapped.done = False
            obs, reward, terminated, truncated, info = env.step(NOOP)
            ffmpeg.stdin.write(obs.tobytes())
            total_frames += 1

    ffmpeg.stdin.close()
    ffmpeg.wait()
    env.close()

    print(f"{outcome} at distance={final_distance}, frames={total_frames}, decisions={decisions}")
    print(f"Saved video: {args.out} ({total_frames / FPS:.1f}s at {FPS}fps)")


if __name__ == "__main__":
    main()
