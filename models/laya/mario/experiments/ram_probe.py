"""Empirically discover NES RAM addresses for enemy screen position in SMB1.

Method: freeze Mario (NOOP every frame) once an enemy is visibly loaded on screen.
With Mario stationary, the camera does not scroll, so the ONLY pixels that change
frame-to-frame belong to the moving enemy (plus minor sprite-animation flicker).
We extract the enemy's pixel centroid from the frame diff (ground truth) and then
scan every RAM byte to find which address's value trajectory best matches that
ground-truth trajectory. This avoids trusting half-remembered address tables.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ.setdefault("USE_TF", "0")

import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A_ONLY, LEFT = range(7)


def find_diff_blob(prev_frame: np.ndarray, frame: np.ndarray, min_pixels: int = 8):
    diff = np.abs(frame.astype(int) - prev_frame.astype(int)).sum(axis=-1)
    mask = diff > 30
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    return {"cx": float(xs.mean()), "cy": float(ys.mean()), "n": int(len(xs)),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    obs, info = env.reset()

    # Walk right until exactly one enemy slot is active, then stop and hold NOOP.
    print("Walking to find a single active enemy...")
    for i in range(400):
        obs, reward, terminated, truncated, info = env.step(RIGHT_A if i % 30 < 4 else RIGHT)
        if terminated or truncated:
            obs, info = env.reset()
        n_active = sum(1 for e in info["enemy_types"] if e != 0)
        if n_active == 1:
            print(f"  found single enemy at frame {i}, x_pos={info['x_pos']}, enemy_types={info['enemy_types']}")
            break
    else:
        print("Never found a clean single-enemy frame; aborting.")
        return

    ram = env.unwrapped.ram
    frames, rams = [obs.copy()], [ram.copy()]
    for _ in range(180):
        obs, reward, terminated, truncated, info = env.step(NOOP)
        frames.append(obs.copy())
        rams.append(ram.copy())
        if terminated or truncated:
            break
    env.close()

    # Ground-truth enemy centroid trajectory from frame diffs.
    # Restrict to the playfield (below the HUD, y>32) and require a sprite-sized blob
    # (an 8x8 or 16x16 NES sprite moving ~1-2px/frame), to reject HUD/animation noise.
    candidates = []  # (frame_idx, cx, cy)
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).sum(axis=-1)
        mask = diff > 30
        mask[:32, :] = False  # HUD
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        if 0 < w <= 20 and 0 < h <= 24 and len(xs) < 200:
            candidates.append((i, float(xs.mean()), float(ys.mean())))

    # Find the longest chain of candidates that move smoothly (small step between nearby frames):
    # this rejects sparse/periodic artifacts (e.g. Mario's own idle animation) in favor of the one
    # object that moves continuously across many frames, which can only be the walking enemy.
    best_chain, cur_chain = [], []
    for k in range(len(candidates)):
        if not cur_chain:
            cur_chain = [candidates[k]]
            continue
        pi, px, py = cur_chain[-1]
        i, x, y = candidates[k]
        if i - pi <= 3 and abs(x - px) < 6 and abs(y - py) < 6:
            cur_chain.append(candidates[k])
        else:
            if len(cur_chain) > len(best_chain):
                best_chain = cur_chain
            cur_chain = [candidates[k]]
    if len(cur_chain) > len(best_chain):
        best_chain = cur_chain

    print(f"Longest smooth-motion chain: {len(best_chain)} frames "
          f"(out of {len(candidates)} size-filtered candidates, {len(frames) - 1} total)")
    valid_idx = [c[0] for c in best_chain]
    gt_x = [c[1] for c in best_chain]
    gt_y = [c[2] for c in best_chain]
    print(f"Usable diff frames: {len(valid_idx)} / {len(frames) - 1}")
    if len(valid_idx) < 8:
        print("Not enough clean diff samples -- enemy likely idle or occluded. Try again.")
        return
    gt_x, gt_y = np.array(gt_x), np.array(gt_y)
    print(f"Ground-truth enemy x range: {gt_x.min():.0f}-{gt_x.max():.0f}, y range: {gt_y.min():.0f}-{gt_y.max():.0f}")

    ram_stack = np.stack([rams[i] for i in valid_idx])  # [T, 2048]

    def best_addresses(target, label, top_k=8):
        errs = np.abs(ram_stack.astype(int) - target[:, None].astype(int)).mean(axis=0)
        order = np.argsort(errs)[:top_k]
        print(f"\nTop {top_k} RAM addresses matching {label} (mean abs error in pixels):")
        for addr in order:
            vals = ram_stack[:, addr]
            print(f"  0x{addr:04X} ({addr:4d})  err={errs[addr]:6.2f}  values={vals[:10].tolist()}")
        return order

    best_addresses(gt_x, "enemy screen-X")
    best_addresses(gt_y, "enemy screen-Y")


if __name__ == "__main__":
    main()
