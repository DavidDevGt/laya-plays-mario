"""Hand-fitted teacher policies for SMB1, in increasing order of information used.

v1 (RuleTeacherV1): only the info-dict fields the RLAgent-based state extractor sees (enemy
count, on_ground). Kept for the baseline comparison table.
v2 (RuleTeacherV2): real RAM-derived enemy distance and tile-ahead ground detection. This is the
"strong teacher" data collection will imitate.
"""


class RuleTeacherV1:
    """Reactive + periodic jump using only info-dict fields (no RAM)."""

    def __init__(self, jump_period: int = 32, hold_decisions: int = 4):
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
        if state["on_ground"] and state["num_enemies_onscreen"] > 0:
            jump = True
        elif state["on_ground"] and self.frames_since_jump >= self.jump_period:
            jump = True
        if jump:
            self.frames_since_jump = 0
            self.hold_remaining = self.hold_decisions - 1
        return "right", jump


class RuleTeacherV2:
    """Reactive jump using real RAM-derived enemy distance, gap-ahead tile detection, and a
    stuck-against-a-wall fallback (pipes/steps read as "solid ground ahead", not a gap, so they
    need their own trigger: if velocity_x stays ~0 while holding right for several decisions in a
    row, something solid is blocking Mario and jumping is the only way past it).

    Jump triggers:
      - a gap is coming up within `gap_lookahead_px` and Mario is on the ground
      - the nearest enemy is within `enemy_jump_dx` ahead (never jump on an enemy *behind* Mario)
      - stuck against an obstacle for `stuck_threshold` consecutive on-ground decisions
    Once triggered, holds the jump button for `hold_decisions` decisions (SMB jump height scales
    with how long A is held).
    """

    def __init__(self, enemy_jump_dx: int = 28, gap_lookahead_px: int = 24, hold_decisions: int = 7,
                stuck_threshold: int = 3, run_on_gap: bool = False, safety_hop_period: int = 0):
        self.enemy_jump_dx = enemy_jump_dx
        self.gap_lookahead_px = gap_lookahead_px
        self.hold_decisions = hold_decisions
        self.stuck_threshold = stuck_threshold
        self.run_on_gap = run_on_gap  # running jumps behave differently (longer/flatter arc);
                                       # the gap-clearing thresholds above were tuned at walking
                                       # speed, so gap jumps drop the run button unless this is set
        self.safety_hop_period = safety_hop_period  # 0 disables; otherwise a small preventive hop
                                       # every N on-ground decisions, so Mario is occasionally
                                       # airborne when an enemy spawns and collides in the very same
                                       # frame it becomes visible in RAM -- unavoidable by reaction
                                       # alone (see runs/logs/phase8_speedrun.md), only by position.
        self.hold_remaining = 0
        self.stuck_count = 0
        self.hold_is_gap = False
        self.frames_since_hop = 0

    def reset(self):
        self.hold_remaining = 0
        self.stuck_count = 0
        self.hold_is_gap = False
        self.frames_since_hop = 0

    def decide(self, state: dict) -> tuple[str, bool, bool]:
        """Returns (move, jump, run)."""
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            run = self.run_on_gap or not self.hold_is_gap
            return "right", True, run

        if state["on_ground"] and state["velocity_x"] == 0:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        jump = False
        is_gap = False
        if state["on_ground"]:
            self.frames_since_hop += 1
            near_enemy = state.get("nearest_enemy_dx") is not None and 0 <= state["nearest_enemy_dx"] <= self.enemy_jump_dx
            gap = state.get("gap_ahead", False)
            stuck = self.stuck_count >= self.stuck_threshold
            safety_hop = self.safety_hop_period > 0 and self.frames_since_hop >= self.safety_hop_period
            jump = near_enemy or gap or stuck or safety_hop
            is_gap = gap and not near_enemy
            if jump:
                self.frames_since_hop = 0
        if jump:
            self.hold_remaining = self.hold_decisions - 1
            self.hold_is_gap = is_gap
            self.stuck_count = 0
        run = self.run_on_gap or not is_gap
        return "right", jump, run
