# Correction to Phase 1: tile-reading vertical offset was wrong

**Date:** 2026-09-21 (same session, discovered during Phase 4 diagnosis)

## What was wrong

`ram_reader.ground_ahead()` used `foot_y = mario_y + 16` to find which tile row to check. This
was validated in `experiments/tile_probe.py` only by checking that it didn't miss a real pit fall
-- it was never checked for false positives (flagging solid ground as a gap). It turned out to be
off by one tile row: a full ASCII dump of the tile grid (16x13) while standing on known-flat
ground showed the real ground row at `sub_y=11`, but `+16` put the query at `sub_y=10` -- one row
of open air *above* the ground, which reads as empty almost everywhere. Measured false-positive
rate on a 159-frame walk across genuinely flat ground: **100%** (`gap_ahead` was True on every
single on-ground frame). Across a full RuleTeacherV2 rollout mixing walking and jumping, the
on-ground false-positive rate was still ~90%.

**Fix:** `foot_y = mario_y + 32` (calibrated directly against the ASCII grid dump). Re-measured
false-positive rate on the same known-flat stretch: **0%**. Re-measured on-ground gap_ahead rate
across a full RuleTeacherV2 rollout: **5.5%** (7/128 on-ground decisions) -- a plausible rate for
a platformer level, not a systematic bug.

## Consequence: the "2754" distance figure in phase1_teacher.md is invalid

RuleTeacherV2's widely-reported 2754 result was achieved *with the buggy detector*. Bizarrely,
the bug (over-triggering "gap ahead" ~90% of the time) made the teacher jump far more often than
warranted, and this over-eager jumping *accidentally* cleared several real obstacles that a more
conservative, correctly-calibrated jump policy does not clear with the same lookahead distance.

With the corrected detector: **RuleTeacherV2 reaches distance 1662** on World 1-1 (gap_lookahead_px
swept over {24,32,40,48,56}; 32-48 all plateau at 1662, so 32 was kept as the new default). Death
cause at 1662 is a suddenly-spawned enemy at close range right after a successful gap-clear and
enemy stomp, not a missed gap -- a genuine limit of a purely reactive policy with no lookahead
beyond what is currently resident in RAM.

## Why this matters for the rest of the pipeline

`teacher_rollouts_v1.jsonl` (4223 decisions) was collected with the buggy detector, so ~45-90% of
its "gap_ahead" state features and a meaningful fraction of its jump labels are contaminated. Both
Laya fine-tunes trained on it (`mario_laya_v1`, `mario_laya_v2_jumponly`) inherited this bad
signal. Data must be re-collected with the fixed `ram_reader.py` before any further fine-tuning is
trustworthy. This is being done next (`teacher_rollouts_v2.jsonl`).

## Lesson

Validating a feature extractor by checking it doesn't miss the *positive* case (recall) is not
enough -- it also has to be checked for false positives (precision) on cases where the answer is
known to be negative. The tile_probe.py fall-precedence test only ever exercised the "should
predict gap" side because it got blocked by a pipe before reaching a real pit.
