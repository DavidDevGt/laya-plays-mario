"""Validate the classic SMB1 tile-grid RAM reading formula (0x500 area map), not just trust it.

Method: walk right holding only RIGHT (never jump) until Mario dies by falling into a pit.
At every frame, use the tile formula to predict whether there is ground directly beneath/ahead
of Mario. If the formula is correct, "no ground ahead" should reliably precede the actual fall
(y_pixel increasing sharply with is_dying) by a small, consistent number of frames.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ.setdefault("USE_TF", "0")

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A_ONLY, LEFT = range(7)


def get_tile(ram, x_level: int, y_pixel: int) -> int:
    """Classic formula: 16x13 tile grid stored at 0x500, split across two half-screen pages."""
    page = (x_level // 256) % 2
    sub_x = (x_level % 256) // 16
    sub_y = (y_pixel - 32) // 16
    if not (0 <= sub_y < 13):
        return -1  # off the visible grid (e.g. above HUD or below screen)
    addr = 0x500 + page * 0xD0 + sub_y * 16 + sub_x
    return int(ram[addr])


def ground_ahead(ram, mario_x: int, mario_y: int, lookahead_px: int = 16) -> bool:
    """Is there a solid tile at foot-level, `lookahead_px` pixels ahead of Mario?"""
    foot_y = mario_y + 16  # sprite is ~16px tall from its top-left anchor; feet near the bottom
    t = get_tile(ram, mario_x + lookahead_px, foot_y)
    return t not in (0, -1)  # 0 == empty air in this map region


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    obs, info = env.reset()
    ram = env.unwrapped.ram

    log = []
    warmup_x = 900  # occasional jumps to clear early enemies/pipes; pure-RIGHT test starts past this x
    for step in range(2500):
        pred_ground = ground_ahead(ram, info["x_pos"], info["y_pixel"])
        in_warmup = info["x_pos"] < warmup_x
        action = (RIGHT_A if step % 20 < 4 else RIGHT) if in_warmup else RIGHT
        obs, reward, terminated, truncated, info = env.step(action)
        log.append({"step": step, "x": info["x_pos"], "y": info["y_pixel"],
                    "pred_ground_ahead": pred_ground, "is_dying": info.get("death", False),
                    "warmup": in_warmup})
        if terminated or truncated:
            print(f"Episode ended at step {step}: death={info.get('death')}, x={info['x_pos']}")
            break

    # Find the fall in the post-warmup (pure-RIGHT, no-jump) phase: first sustained y increase
    # (falling below the visible screen) that ends in death, not a normal single-frame jitter.
    post = [e for e in log if not e["warmup"]]
    fall_step = None
    for i in range(1, len(post)):
        if post[i]["y"] - post[i - 1]["y"] > 15:
            fall_step = post[i]["step"]
            break

    if fall_step is None:
        print("No pit fall detected in the pure-RIGHT phase (died to enemy or ran out of steps).")
        print(f"Ran {len(log)} steps, final x={log[-1]['x']}, died={log[-1]['is_dying']}")
        return

    idx = next(i for i, e in enumerate(log) if e["step"] == fall_step)

    print(f"\nFall detected at step {fall_step} (x={log[idx]['x']}, y={log[idx]['y']})")
    print("Predicted ground-ahead in the 15 steps before the fall:")
    for i in range(max(0, idx - 15), idx + 2):
        marker = " <-- FALL" if i == idx else ""
        print(f"  step {log[i]['step']:4d} x={log[i]['x']:4d} y={log[i]['y']:3d} "
              f"pred_ground_ahead={log[i]['pred_ground_ahead']}{marker}")

    env.close()


if __name__ == "__main__":
    main()
