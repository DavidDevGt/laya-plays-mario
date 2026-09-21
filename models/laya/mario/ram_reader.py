"""SMB1 RAM reading, empirically validated against ground truth (see experiments/ram_probe.py,
experiments/tile_probe.py) rather than trusted from memory alone.

Confirmed by:
  - Enemy X/Y: structural adjacency to the package's own verified player-position addresses
    (0x86 player screen-x, 0x6d player page) sits immediately before the enemy arrays (0x87, 0x6E),
    plus the enemy Y array (0xCF+slot) matches the pixel ground-truth trajectory from frame-diffing
    almost exactly once a +16px sprite-anchor offset is applied (184 + 16 == 200, measured).
  - Tile grid: validated in experiments/tile_probe.py by checking that "no ground ahead" precedes
    an actual pit fall.
"""
from typing import Optional

_ENEMY_TYPE_ADDR = 0x16       # +slot, slot in [0, 5)
_ENEMY_DRAWN_ADDR = 0x0F      # +slot: nonzero if the slot is actively rendered
_ENEMY_PAGE_ADDR = 0x6E       # +slot: high byte of enemy level-x
_ENEMY_SCREEN_X_ADDR = 0x87   # +slot: low byte of enemy level-x (== on-screen x within its page)
_ENEMY_SCREEN_Y_ADDR = 0xCF   # +slot: enemy y, needs +16 to match on-screen pixel y
_ENEMY_Y_OFFSET = 16

_PLAYER_PAGE_ADDR = 0x6D
_PLAYER_X_ADDR = 0x86

N_ENEMY_SLOTS = 5


def enemy_slots(ram) -> list[dict]:
    """All currently-active enemy slots with absolute level position and type.

    `_ENEMY_DRAWN_ADDR` alone marks a slot active -- type 0 is a real enemy class (Green Koopa
    Troopa), not "empty". An earlier version also filtered on `etype == 0`, which silently made
    the whole pipeline blind to every Green Koopa on the level; see runs/logs/phase8_speedrun.md.
    """
    out = []
    for slot in range(N_ENEMY_SLOTS):
        if int(ram[_ENEMY_DRAWN_ADDR + slot]) == 0:
            continue
        etype = int(ram[_ENEMY_TYPE_ADDR + slot])
        page = int(ram[_ENEMY_PAGE_ADDR + slot])
        x = page * 256 + int(ram[_ENEMY_SCREEN_X_ADDR + slot])
        y = int(ram[_ENEMY_SCREEN_Y_ADDR + slot]) + _ENEMY_Y_OFFSET
        out.append({"slot": slot, "type": etype, "x": x, "y": y})
    return out


def player_abs_x(ram) -> int:
    return int(ram[_PLAYER_PAGE_ADDR]) * 256 + int(ram[_PLAYER_X_ADDR])


def nearest_enemy(ram, player_x: int, player_y: int) -> Optional[dict]:
    slots = enemy_slots(ram)
    if not slots:
        return None
    for e in slots:
        e["dx"] = e["x"] - player_x
        e["dy"] = e["y"] - player_y
    return min(slots, key=lambda e: abs(e["dx"]))


def get_tile(ram, x_level: int, y_pixel: int) -> int:
    """16x13 tile grid at 0x500, split across two half-screen pages of 16x13 = 208 (0xD0) bytes."""
    page = (x_level // 256) % 2
    sub_x = (x_level % 256) // 16
    sub_y = (y_pixel - 32) // 16
    if not (0 <= sub_y < 13):
        return -1
    addr = 0x500 + page * 0xD0 + sub_y * 16 + sub_x
    return int(ram[addr])


_FOOT_Y_OFFSET = 32  # calibrated empirically: mario_y (~176 while standing) + 16 lands one tile
                     # row ABOVE the actual ground row; +32 lands inside the true ground row.
                     # See experiments/tile_probe.py ASCII-grid dump for the calibration check.


def ground_ahead(ram, player_x: int, player_y: int, lookahead_px: int) -> bool:
    t = get_tile(ram, player_x + lookahead_px, player_y + _FOOT_Y_OFFSET)
    return t not in (0, -1)
