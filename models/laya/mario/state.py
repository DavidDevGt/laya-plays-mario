"""Structured state extraction for SMB1: RAM (validated, see ram_reader.py) + emulator info dict.

This is the single source of truth for what both the rule teacher and Laya see. Keeping it in one
place (rather than duplicating field logic per-policy) is what makes the pipeline's imitation data
consistent between teacher and learner.
"""
from . import ram_reader

GAP_LOOKAHEAD_PX = 24  # default; override via MarioStateExtractor(gap_lookahead_px=...) to match
                       # whatever a given teacher/policy was tuned against. 24 is what
                       # RuleTeacherV2's walking-speed level clear was tuned and verified against.


class MarioStateExtractor:
    def __init__(self, history_len: int = 5, gap_lookahead_px: int = GAP_LOOKAHEAD_PX):
        self.history_len = history_len
        self.gap_lookahead_px = gap_lookahead_px
        self.prev_x = None
        self.prev_y = None
        self.action_history: list[str] = []

    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self.action_history = []

    def record_action(self, action_name: str):
        self.action_history.append(action_name)
        self.action_history = self.action_history[-self.history_len:]

    def build(self, ram, info: dict) -> dict:
        x = ram_reader.player_abs_x(ram)
        y = info["y_pixel"]
        vx = 0 if self.prev_x is None else x - self.prev_x
        vy = 0 if self.prev_y is None else y - self.prev_y
        self.prev_x, self.prev_y = x, y

        nearest = ram_reader.nearest_enemy(ram, x, y)
        gap = not ram_reader.ground_ahead(ram, x, y, self.gap_lookahead_px)

        return {
            "mario_x": x,
            "mario_y": y,
            "velocity_x": vx,
            "velocity_y": vy,
            "on_ground": vy == 0,
            "nearest_enemy_dx": nearest["dx"] if nearest else None,
            "nearest_enemy_dy": nearest["dy"] if nearest else None,
            "nearest_enemy_type": nearest["type"] if nearest else None,
            "num_enemies_onscreen": len(ram_reader.enemy_slots(ram)),
            "gap_ahead": gap,
            "time_remaining": info["time"],
            "recent_actions": self.action_history[-3:],
        }
