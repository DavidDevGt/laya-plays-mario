# Phase 8: pushing the teacher to a full level clear

Goal: complete World 1-1, as fast as possible. Started from RuleTeacherV2 at distance 1662
(Phase 1 corrected result).

## Bug #3: enemy type 0 was silently filtered out as "no enemy"

`ram_reader.enemy_slots()` had `if etype == 0: continue`, added under the (unverified) assumption
that type 0 meant an empty/inactive slot. Frame-by-frame RAM inspection at the exact death point
(x=1662) showed slot 2 with `drawn=1` (active) at x=1669, only 7px ahead of Mario, for at least 3
frames before the fatal collision -- fully visible in RAM, just discarded by this filter. Type 0
is Green Koopa Troopa, a real, common enemy in 1-1's second half. This bug made the entire
pipeline (teacher, all three data collections, all four Laya fine-tunes) blind to every Green
Koopa on the level, all session.

Fix: drop the `etype == 0` check; `drawn` alone marks a slot active (matches how
`_ENEMY_DRAWN_ADDR` is used everywhere else). Distance: 1662 -> **2014**.

## Bug fix #4 (parameter, not code): hold_decisions was too short for a multi-enemy gauntlet

Past x=2014 the level has a run of 3-5 Goombas clustered together over a gap (the section right
before the final staircase). `hold_decisions=5` (jump held ~5 decisions = ~20 frames) wasn't
enough hang time to clear the whole cluster in one jump; the trigger would fire, Mario would start
descending mid-cluster, and land on/into a Goomba. Swept `hold_decisions` in {6,7,8,9}: 6 reaches
2469 and still dies; **7 clears the entire level** (`flag_get=True`, distance=3161, in-game
time_used=110/400). 8 and 9 give identical results to 7 (no further benefit, extra hang time past
what's needed). New default: `hold_decisions=7` (was 5).

## Running (B button) was tried and reverted

Holding B (`run=True`) gives higher top speed and longer jump arcs in SMB1, which would directly
serve "as fast as possible". Tried three variants:
1. Running the whole time: fails at distance ~1128-1155 regardless of threshold retuning --
  running jumps have different arc/timing than the walking-speed jumps this teacher's thresholds
  were calibrated against, and re-tuning by grid search plateaued without matching the walking
  teacher's distance.
2. Hybrid (run on flat ground, drop to walk only during the jump itself): same failure. In SMB1,
  horizontal velocity built up during a running approach doesn't reset just because B is released
  mid-air -- the momentum is already "locked in" from before the jump, so toggling run only at the
  jump doesn't reproduce true walking-speed jump physics.
3. Finer `action_repeat` (faster reaction cadence): also regressed, because `hold_decisions` and
  `stuck_threshold` are calibrated in units of *decisions*, not frames -- changing action_repeat
  changes how many real frames a "decision" spans and desyncs every existing threshold.

**Decision: kept walking speed (run=False) for the verified clear** rather than keep sinking time
into re-deriving running-jump physics constants from scratch. This is a legitimate scope cut, not
a dead end -- documented here so a future pass knows where the recalibration work would start
(distance ~1128, running-jump arc height/length as a function of run-up frames) rather than
re-discovering it.

## Final verified result

`RuleTeacherV2(enemy_jump_dx=28, gap_lookahead_px=32, hold_decisions=7, stuck_threshold=3)`,
walking speed, World 1-1: **flag_get=True, distance=3161, in-game time used 110s (of 400s
budget), 2210 emulator frames, 553 decisions, deterministic/reproducible** (confirmed by
re-running).

Recorded to `runs/videos/teacher_v2_clear.mp4` (36.8s real-time replay at 60fps).

## Not yet done (natural next steps, not attempted this pass)

- Re-run data collection (Phases 3) and Laya fine-tuning (Phases 4/5/7) with this fixed,
  level-clearing teacher -- all existing Laya checkpoints were trained on data from a
  Green-Koopa-blind, non-clearing teacher, so they inherited that blind spot too.
  Re-collecting now gives a strictly better imitation target.
- Running-speed retuning (see above) to actually reduce the 110s clear time.
- Extending past World 1-1 to the other 7 collection stages, several of which the teacher has
  never been checked against past their original (buggy) death points either.
