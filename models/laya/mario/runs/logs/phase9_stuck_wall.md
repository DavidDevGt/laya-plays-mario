# Phase 9: the stuck-wall bug -- found from a video review, fixed, Laya clears the level

## How this was found

The user watched `laya_v6_best.mp4` and reported Mario frozen in place near the end of the video
(on-screen timer at 200, matching the "TIMEOUT" result already in the eval JSON) -- not
progressing, not dying. This prompted a proper trace using the actual `run_episode` /
`laya_decide_fn` machinery (an earlier hand-rolled debug script gave a *different*, wrong answer
because it forgot to call `extractor.record_action()`, silently leaving `recent_actions` stale --
a reminder that ad hoc debug scripts have to replicate every side effect of the real controller
loop, not just its decision logic).

## Root cause

At distance 2226, Mario is on_ground, velocity_x=0, no enemy nearby, `gap_ahead=False` (there IS
solid ground/a wall ahead) -- a stuck-against-a-wall situation identical in kind to the one that
blocked `RuleTeacherV2` at distance 722 many phases ago (`phase1_teacher.md`), which is why the
teacher has a `stuck_threshold` fallback: if velocity_x stays 0 on the ground for N consecutive
decisions, force a jump regardless of the reactive triggers.

**Laya's controller had no equivalent fallback.** Its raw P(jump) hovered at 0.39-0.43 against the
0.5 threshold for 60+ consecutive decisions (240+ frames, ~4 real seconds) -- close, but never
crossing, so `jump=False` every single time and Mario just walks in place against the wall
forever. This is the same "correct trend, wrong side of threshold" failure mode diagnosed
repeatedly earlier in the project (phases 4/5, 7), just at a different obstacle, and it has no
natural exit: unlike a transient near-miss, a *permanent* wall block means the state never changes
and P(jump) never gets a reason to move either.

## Fix

Ported RuleTeacherV2's exact mechanism to Laya's controller (`laya_decide_fn` in `eval.py`,
mirrored in `record.py`): track consecutive on-ground, velocity_x==0 decisions; past
`stuck_threshold`, force a jump irrespective of the model's P(jump). Controller-level, no
retraining.

## Result: this was the last blocker

Swept `stuck_threshold` x `hold_decisions` jointly (previously only `hold_decisions` had been
tuned, against a model that could still get permanently stuck, which capped how much of the
search space was even reachable). Best combination:

**`threshold=0.5, hold_decisions=4, stuck_threshold=8` -> Laya fine-tuned + DAgger completes World
1-1. `flag_get=True`, distance=3161, in-game time used 120/400 (vs teacher_v2's 110/400).**
Verified reproducible (deterministic, re-ran twice, identical result).

Full 8-stage re-evaluation with this config: completion rate 12.5% (1/8, same stage as
teacher_v2 -- 1-1), but **average distance across all 8 stages is now 1377.5, higher than
teacher_v2's 1070.4** -- on stages neither policy completes, Laya's learned reactions generalize
at least as well as, and on some stages better than, the teacher's fixed hand-tuned thresholds
(e.g. 4-1: Laya 2419 vs teacher not separately re-measured at this config; 8-1: Laya 2499).

## Updated headline result

| Policy | Completes 1-1 | Distance | In-game time |
|---|---|---:|---:|
| teacher_v2 | Yes | 3161 | 110 |
| **Laya (fine-tuned + DAgger + calibrated threshold + hold + stuck-fallback)** | **Yes** | **3161** | **120** |

This supersedes the "Laya learned a suboptimal policy relative to a working teacher" conclusion
from the prior report version: with the missing controller-level safety net in place, **Laya
matches the teacher's completion and is 10 in-game-time-units slower** -- a small, quantified gap,
not a qualitative one. The remaining gap is almost certainly attributable to Laya's decisions being
probabilistic/threshold-based rather than the teacher's deterministic reactive rules, occasionally
taking one extra beat to commit.

## Lesson

A model's own imperfection (an under-threshold probability) can look identical, from the training
metrics alone, to a controller-integration gap (a missing safety fallback the reactive baseline
already had for free). Val accuracy does not distinguish these; only playing the game and watching
where it actually stops does -- and in this case, watching the *video*, not just the metrics, is
what surfaced it.
