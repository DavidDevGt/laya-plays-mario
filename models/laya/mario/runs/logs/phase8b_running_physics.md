# Phase 8b: running (B button) physics investigation -- documented failure

## Bug #5 found and fixed along the way

`gap_lookahead_px` on `RuleTeacherV2` was a dead parameter: `MarioStateExtractor.build()` computed
`gap_ahead` using a hardcoded module-level constant (`GAP_LOOKAHEAD_PX = 24` in `state.py`),
completely ignoring whatever the teacher was constructed with. Every `gap_lookahead_px` sweep run
earlier in this session (Phases 1, 7, 8) was silently a no-op on that parameter -- any behavior
differences seen were from the *other* parameters that were varied alongside it in the same test
tuples. Confirmed by isolating just `gap_lookahead_px` post-fix: it now visibly changes outcomes.

Fixed: `MarioStateExtractor.__init__` now takes `gap_lookahead_px` and uses it instead of the
module constant. Callers must construct the extractor with the same value the teacher uses (both
default to 24 now, so `MarioStateExtractor()` + `RuleTeacherV2()` stay in sync by default).

Re-verified the walking clear with the *actually*-wired parameter: **`gap_lookahead_px=24`
(not 32, which was the pre-fix "working" value by coincidence of the bug) + `hold_decisions=7`
still clears the level in 110 in-game seconds.** This is the final, confirmed teacher_v2 config.

## Running attempt: systematic, did not reach completion

Hypothesis: holding B (run) raises max speed (~12-13 px/frame vs ~7 walking) and jump distance,
which should let a reactive policy finish faster. Tried, in order:

1. Run the whole time, walking-calibrated thresholds: fails at distance ~1128-1155.
2. Hybrid (drop run only during the jump itself): same failure -- momentum from the run-up carries
  into the jump regardless of whether B is held during the jump.
3. Faster `action_repeat` (finer reaction granularity): regressed further -- `hold_decisions` and
  `stuck_threshold` are counted in decisions, not frames, so this silently rescales every
  threshold.
4. **After fixing bug #5**, re-swept `gap_lookahead_px` x `hold_decisions` properly for running:
  best found is `gap_lookahead_px=45-50, hold_decisions=8` -> **distance 1961/3260 (~60%
  of the level), still no flag.** Higher hold_decisions (9, 10, 12) did not improve further
  (non-monotonic: 1433, 706, 1125) -- there is no single global (lookahead, hold) setting that
  clears the level at running speed with this reactive architecture.

## Root cause (best available explanation, not fully reverse-engineered)

SMB1's jump physics are governed by discrete velocity/gravity lookup tables indexed by Mario's
speed state, not continuous physics -- a "running jump" does not simply extend a "walking jump"'s
arc proportionally to hold duration. Different obstacles on the level apparently need different
(lookahead, hold) combinations to clear at running speed, and no single global setting found so
far satisfies all of them simultaneously, whereas one walking-speed setting does. Precisely
modeling the running-jump arc (as was done for the tile grid and enemy RAM addresses) would need
its own dedicated empirical calibration pass (e.g., frame-by-frame position tracking of a running
jump at several hold durations, fit against the actual velocity/gravity tables) -- not attempted
here due to time budget; flagged as future work.

## Decision

**RuleTeacherV2 walking-speed config (`gap_lookahead_px=24, hold_decisions=7, enemy_jump_dx=28,
stuck_threshold=3`) is the locked-in "teacher_v2" for the rest of this project.** It is the only
configuration found (walking or running) that reliably completes World 1-1. Running is documented
here as an investigated-but-unresolved speed optimization; per the project brief's request not to
hide failed hypotheses, this is reported as exactly that: a real attempt, a real partial result
(60% clear at running speed), and a concrete reason it wasn't pursued further within budget.

This also means: any additional speedrun-time optimization work should target the *walking*
teacher and Laya's imitation of it (trimming unnecessary jumps, tightening thresholds without
breaking the clear, possibly a smarter/adaptive B-tap that only engages on provably-safe
straightaways once those are identified by successful walking traversal) rather than a uniform
run-vs-walk toggle.
