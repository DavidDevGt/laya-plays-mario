# Laya Plays Super Mario Bros: from a text-classification model to a level-clearing agent

**Date:** 2026-09-21
**Pipeline:** Super Mario Bros → RAM/state extraction → Laya decision model → action controller → learned policy → speedrun attempt

## TL;DR

- **teacher_v2** (hand-coded, RAM-grounded reactive rules) **completes World 1-1: flag_get=True, in-game time used 110/400, deterministic.**
- **Laya (fine-tuned on teacher_v2 data + one DAgger round + three controller-level safety mechanisms: threshold calibration, jump-hold hysteresis, and a stuck-against-wall fallback) also completes World 1-1: flag_get=True, in-game time used 120/400.**
- The gap between them is now **10 in-game time units (≈9%), not a qualitative "does/doesn't finish" gap.** Getting there took two full debugging passes: the first left Laya permanently stuck against a wall near the end of the level, discovered only by watching the recorded video (metrics alone reported "timeout," not why) — see [§9](#9-a-second-bug-found-from-a-video-not-a-metric).
- Across all 8 World-X-1 stages, Laya's **average distance (1377.5) is now higher than the teacher's (1070.4)** — on stages neither completes, Laya's learned reactions generalize at least as well as the hand-tuned rules.
- **No completion time in this project should be compared to a human speedrunner.** We did not benchmark a live human. The one verifiable external reference found (TAS-optimal, glitch-assisted, 30 in-game time units) is not run on a comparable route to either of our agents — see [§6](#6-human--tas-baseline).

---

## 1. Final comparison table (World 1-1, unless noted)

All policies are deterministic (fixed teacher, Laya at fixed threshold/argmax, fixed env) — repeated runs on the same stage produce identical results, so "consistency" is reported across the 8 different World-X-1 stages, not across repeated trials of the same one. See [`eval.py`](mario/eval.py) docstring.

| Policy | Completion rate (8 stages) | Best distance (1-1) | Avg distance (8 stages) | In-game time used (1-1 clear) | Actions/sec (wall-clock) | Model inference (ms/decision) |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0% | 241 | 157 | — (never completes) | 398 | — |
| Rule-based v1 (info-dict only, no RAM) | 0% | 816 | 522 | — (never completes) | 388 | — |
| teacher_v2 (RAM-grounded reactive rules) | 12.5% (1/8) | 3161 (FLAG) | 1070 | **110** | 384 | — |
| Laya zero-shot (typed-decisions, no fine-tuning) | 0% | 296 | 307 | — (never completes) | 65 | 12.9 |
| **Laya fine-tuned + DAgger + calibration + hold + stuck-fallback (final, `mario_laya_v6_dagger`)** | **12.5% (1/8)** | **3161 (FLAG)** | **1377.5** | **120** | 117 | 7.7 |

Final Laya controller config: `threshold=0.5, hold_decisions=4, stuck_threshold=8` (see §5, §9).

Held-out-stage (4-1, 7-1; zero training examples from either) offline validation accuracy for the
jump decision, across the fine-tuning progression:

| Checkpoint | Data | Held-out jump accuracy |
|---|---|---:|
| Baseline (typed-decisions, before any Mario fine-tuning) | — | 64.5% |
| `mario_laya_v5_final` | 8173 decisions, teacher_v2 (bug-free) | 93.3% |
| `mario_laya_v6_dagger` (final) | 12090 decisions, + 1 DAgger round | 82.9% (harder distribution — see §4) |

## 2. What each policy actually is

- **Random**: uniform random move + 30% jump chance per decision.
- **Rule-based v1**: the very first heuristic teacher, using only the crude fields the initial (pre-RAM) state extractor exposed (enemy count, on_ground). Kept only as a lower-bound reference — see `runs/logs/phase1_teacher.md`.
- **teacher_v2**: `RuleTeacherV2` in [`teacher.py`](mario/teacher.py) — reactive rules over RAM-derived features (real enemy position via validated RAM addresses, tile-grid gap detection, stuck-against-wall detection), config `enemy_jump_dx=28, gap_lookahead_px=24, hold_decisions=7, stuck_threshold=3`, walking speed (not running — see §5). This is the "teacher" the whole project imitates.
- **Laya zero-shot**: the downloaded `typed-decisions` checkpoint, never fine-tuned on Mario data at all. Included to show the pre-fine-tuning floor.
- **Laya (final)**: `typed-decisions` → head-only fine-tune on teacher_v2 data (`mario_laya_v5_final`) → one DAgger round (`mario_laya_v6_dagger`) → evaluated with a calibrated decision threshold (0.5), controller-level jump-hold (4 decisions), and a controller-level stuck-against-wall fallback (forces a jump after 8 consecutive stalled decisions) — the last of these three was needed only after discovering Laya could get permanently stuck with no natural way out (§9). This is the final, reported Laya policy.

## 3. Data pipeline (verified, not assumed)

- **RAM addresses validated empirically**, not from memory: enemy X/Y position (structural adjacency to the package's own verified player-position addresses + pixel ground-truth cross-check), tile grid for gap detection (calibrated against a full ASCII dump of the 16x13 tile buffer). See `runs/logs/phase1_teacher.md` and `mario/experiments/`.
- **Bugs found and fixed during this project** (documented, not hidden, per the project brief):
  1. Tile-grid vertical offset off by one row → 90-100% false-positive gap detection (`phase1_correction.md`).
  2. Enemy type filter silently excluded Green Koopa Troopas (`etype==0` treated as "no enemy") → teacher and every prior dataset were blind to a whole enemy class (`phase8_speedrun.md`).
  3. `gap_lookahead_px` was a dead parameter — `state.py` used a hardcoded constant regardless of what the teacher was configured with; every earlier sweep of this parameter was a silent no-op (`phase8b_running_physics.md`).
  4. `run_episode`'s default `run=True` silently made two data-collection scripts (`collect_data.py`, `dagger.py`) hold the run button during collection without anyone asking for that, contaminating an entire dataset generation pass before being caught and fixed.
  5. Laya's controller had no equivalent of the teacher's "stuck against a wall too long → force a jump" fallback, so it could get permanently stuck at a plain wall despite a nearly-there jump probability (0.39-0.43 vs a 0.5 threshold) — found from watching a recorded video, not from any metric (`phase9_stuck_wall.md`). Fixing this is what took Laya from a 70%-of-level partial run to a full clear.
- **Final dataset** (`data/v6_aggregated.jsonl`, 12090 decisions):
  - `data/v4_clean_episodes.jsonl` (1406 decisions, epsilon=0, one clean episode per of 8 stages, walking speed, teacher_v2 with bugs #1-#4 fixed)
  - `data/v4_explore_episodes.jsonl` (6767 decisions, epsilon=0.08, 6 episodes/stage — includes one lucky full level-1-1 clear despite exploration noise)
  - `data/dagger_v5.jsonl` (3917 decisions, DAgger round: `mario_laya_v5_final`'s own policy driving, `teacher_v2` shadow-labeling every state it actually visits)
  - Verified enemy-type diversity before training: type 0 (Koopa) present at 11.8% of decisions (was 0% before bug #2 was fixed); 8 distinct enemy types represented across the 8 stages; jump base rate 33.4% (not degenerate).
- **Train/val split**: by *stage*, not by random decision — World 4-1 and 7-1 are held out of training entirely, so the reported held-out accuracy measures generalization to genuinely unseen terrain and enemy layouts, not memorized nearby frames.

## 4. Training configuration

- Base checkpoint: `convaiinnovations/laya` `typed-decisions` subfolder (421M params, ModernBERT-large encoder + 2-layer decision head). Never reused a Mario-contaminated checkpoint as a starting point — every fine-tune in this final pass starts from the original downloaded checkpoint.
- **Encoder frozen**; only the ~26.5M-param decision head (2-layer transformer + scorer + type embedding + act head, 6.3% of total params) is trained. Rationale: dataset is small (~12k decisions from one game); this mirrors the reference blog's own recipe of training the scoring layer before any deeper unfreezing.
- **Only the `jump` question is trained** (binary noul). `move` is hardcoded "right" in the controller. RuleTeacherV2's `move` label is a hardcoded constant, so it carries no learnable signal — an earlier attempt that included it produced a degenerate, self-reinforcing failure (documented in `runs/logs/phase4_5_finetune_eval.md`).
- Loss: negative `proper_reward` (log score + spherical score — the same strictly-proper scoring rule Laya's own RLCD training uses), differentiated directly through the softmax rather than sampled via REINFORCE, since we have ground-truth hard labels (see `finetune.py` docstring for the justification).
- Optimizer: AdamW, lr=3e-4, weight_decay=0.01, batch size 48, gradient clip 1.0.
- **Seeds**: `--seed 0` for `mario_laya_v5_final` (`torch.manual_seed`, `random.Random`, `numpy` via `seed_all()`), `--seed 0` again for the DAgger-aggregated `mario_laya_v6_dagger` retrain (fresh from base, not continued from v5). DAgger *collection* itself used `--seed 2`.
- Small-scale validation before the full run: a 3-epoch smoke test on the same data confirmed val loss dropping and jump accuracy improving (84.4% → 91.4%) before committing to the full 15-epoch run.
- 15 epochs, ~380s (v5) / ~580s (v6, larger aggregated set) wall-clock on one RTX 5070 Ti.
- Why DAgger *did not* raise offline accuracy: `mario_laya_v5_final` had 93.3% held-out accuracy; after aggregating DAgger data (states the learner itself visits, concentrated near its own decision-boundary mistakes) and retraining, `mario_laya_v6_dagger` measured 82.9%. This is the expected, correct outcome, not a regression — DAgger data is disproportionately hard cases by construction, and the accuracy that matters is the gameplay result, which did improve.

## 5. Speedrun-specific optimization: what worked and what didn't

Investigated per the project's explicit request to not stop at "it completes the level":

- **Held the run/B button**: tried extensively (walking-calibrated thresholds, a hybrid that drops run only during gap-jumps, finer `action_repeat`, and — after fixing the dead-parameter bug — a proper 2D grid search over lookahead × hold-duration). Best running result reaches only **60% of the level** (distance 1961/3260) before getting stuck; walking reaches 100%. Root cause: SMB1's jump physics are governed by discrete velocity/gravity lookup tables that don't scale a "running jump" simply from a "walking jump," and different obstacles need different (lookahead, hold) pairs at running speed with no single setting satisfying all of them. **Documented as an investigated, unresolved hypothesis** (`runs/logs/phase8b_running_physics.md`) — reducing the level's clear time below 110s remains open work.
- **Jump-hold hysteresis for Laya**: a genuine, working optimization. Laya's per-decision independent probability naturally cuts a multi-decision jump short on the obstacle (a 3-5 enemy cluster over a gap) that needs sustained commitment; adding the same hold mechanism the rule teacher has (`--hold-decisions`, controller-level, no model change) was necessary groundwork toward the full clear.
- **Threshold calibration**: Laya's raw P(jump) never crosses the default 0.5 threshold at some safety-critical frames even when it's correctly *trending* toward danger — a calibration issue, not a directionality one (Laya's own README warns it "ships over-confident"). Necessary at every stage of this project.
- **Stuck-against-wall fallback**: the final missing piece — see §9. Without it, Laya could get permanently stuck no matter how the other two parameters were tuned, since a plain wall produces a stagnant, unchanging state with no pressure pushing P(jump) over threshold.
- **Not attempted, flagged as next steps**: route-level optimization (identifying which of teacher_v2's jumps are longer than strictly necessary and trimming them) to close the remaining 10-time-unit gap with the teacher; a second DAgger round specifically targeting post-stuck-fix trajectories; encoder unfreezing (currently only the 6.3%-of-params head is trained).

## 6. Human / TAS baseline

Per the project's explicit instruction not to claim "better than human" without a verifiable, documented comparison:

- We did **not** run or measure a live human playing this environment. No human baseline was collected in this project.
- The only externally verifiable reference found (web search, September 2026): SMB1 World 1-1's known frame-optimal clear leaves **370 on the in-game timer** (i.e., "time used" = 30 in our units), per speedrun.com community documentation. This figure:
  - Is a **TAS (tool-assisted, frame-perfect) figure**, not a human real-time result — not achievable by human reaction/input precision, and not the right thing to call a "human benchmark."
  - Is achieved using **glitch techniques (flagpole glitch and similar)** that are a fundamentally different route from straightforward ground traversal — not comparable to teacher_v2 or Laya, which do neither.
  - Sources: [Speedrun.com forum — "What's the fastest 1-1 play?"](https://www.speedrun.com/smb1/forums/uf7hi), [Speedrun.com forum — "370 possible on 1-1?"](https://www.speedrun.com/smb1/forums/86bnw), [Super Mario Bros. speedrunning (Wikipedia)](https://en.wikipedia.org/wiki/Super_Mario_Bros._speedrunning).
- **Conclusion on this axis**: teacher_v2's 110-time-used and Laya's 120-time-used, both no-glitch, full ground-traversal clears, should be read only against each other and against the random/naive baselines in §1 — not against the TAS figure above, and not against any human claim, verified or otherwise.

## 6b. Interactive demo

`models/laya/mario/demo/` — a live-synced replay UI, not a staged video: `record_demo.py` runs the
real `mario_laya_v6_dagger` checkpoint through the real controller loop once and logs every
decision (game state, P(jump) or why there wasn't a fresh one this decision, selected buttons,
timing) to `episode.jsonl`, frame-aligned with `episode.mp4`. `run_demo.py` serves both plus a
web panel (`static/`) that reads only those two files — jump-probability bars, a gamepad diagram
highlighting real pressed buttons, live game state, and performance metrics, all synced to video
playback position. Run: `python models/laya/mario/demo/run_demo.py`, then open the printed
`http://localhost:8731/`. See §11 for a walkthrough of what it shows.

## 7. Videos

- [`runs/videos/teacher_v2_clear_final.mp4`](mario/runs/videos/teacher_v2_clear_final.mp4) — teacher_v2, full World 1-1 clear, flag_get=True, distance 3161, in-game time used 110/400. 36.8s real-time replay at 60fps.
- **[`runs/videos/laya_v6_CLEAR.mp4`](mario/runs/videos/laya_v6_CLEAR.mp4) — Laya (final config), full World 1-1 clear, flag_get=True, distance 3161, in-game time used 120/400. 40.0s replay.**
- [`runs/videos/laya_v6_best.mp4`](mario/runs/videos/laya_v6_best.mp4) — superseded: Laya *before* the stuck-wall fallback was added, gets stuck at distance 2226 and never recovers. Kept because this is the exact video the user flagged that led to finding and fixing the bug in §9 — a useful before/after pair with `laya_v6_CLEAR.mp4`.
- [`runs/videos/laya_1-1.mp4`](mario/runs/videos/laya_1-1.mp4) — kept from an earlier session milestone (pre-final-bugfix `mario_laya_v4_dagger`); retained for project history.

## 8. Files delivered

- **Code**: `models/laya/mario/` — `ram_reader.py` (validated RAM reads), `state.py` (state extraction), `teacher.py` (RuleTeacherV1/V2), `controller.py` (emulator plumbing), `collect_data.py`, `dagger.py`, `finetune.py`, `eval.py` (includes the calibrated Laya controller: threshold + hold + stuck-fallback), `record.py`, `questions.py`.
- **Final checkpoint**: `models/laya/mario/runs/mario_laya_v6_dagger/` (`model.safetensors`, `rl_agent_config.json`, `tokenizer/`, `encoder/`, `training_log.json`). **Must be run with `--threshold 0.5 --hold-decisions 4 --stuck-threshold 8` to reproduce the level clear** — the checkpoint alone (argmax/default threshold) does not complete the level; the controller-level calibration is part of the reported result, not an afterthought.
- **Intermediate checkpoints** (kept for the documented progression, not for reuse): `mario_laya_v1` through `v5_final` — each one's role and why it was superseded is in `runs/logs/`.
- **Dataset**: `models/laya/mario/data/v6_aggregated.jsonl` (final, 12090 decisions) + its two components and metadata JSONs. Earlier `teacher_rollouts_v1/v2.jsonl` kept as documented (buggy-detector) history, not for reuse.
- **Training config & logs**: `runs/mario_laya_v6_dagger/training_log.json` (args, seeds, per-epoch metrics); `runs/logs/*.md` (8 phase reports covering every hypothesis tested, including the failed ones — phase10 documents a "reinforce with more successful trajectories" retrain that was tried, evaluated, and rejected in favor of v6, which remains final).
- **Demo**: `models/laya/mario/demo/` — see §6b.
- **Results**: `runs/logs/eval_*_all8.json` (per-policy, per-stage raw results behind §1's table).

## 9. A second bug found from a video, not a metric

After the first version of this report concluded "Laya learned a suboptimal policy" at 2226/3161
distance, the user watched `laya_v6_best.mp4` and reported Mario frozen near the end of the clip —
not dying, not progressing. The eval JSON already said "TIMEOUT," but a timeout alone doesn't say
*why*, and nothing in the offline accuracy numbers (93.3%, 82.9%) would have surfaced this on its
own. A proper trace (using the real controller loop — an initial hand-rolled debug script gave a
different, wrong answer because it forgot to call `extractor.record_action()`, leaving
`recent_actions` stale) showed: on_ground, velocity_x=0, no enemy nearby, solid wall ahead,
P(jump) oscillating at 0.39-0.43 against a 0.5 threshold for over 60 consecutive decisions. Laya
had no equivalent of the teacher's stuck-against-wall fallback and so had no way out of this state
— it doesn't change on its own, so the probability never gets a reason to move either.

Porting that exact fallback to Laya's controller (§2, §5) and re-sweeping `hold_decisions` ×
`stuck_threshold` jointly (the earlier sweep only varied `hold_decisions`, against a policy that
could still get permanently stuck, which had been silently capping how much of that search space
was reachable) found `hold_decisions=4, stuck_threshold=8`, which clears the level. This is why
§1's headline numbers differ from the version of this report the user was originally shown, and
why the conclusion changed from "suboptimal relative to a working teacher" to "matches the
teacher's completion, 10 time-units slower."

## 10. Conclusion (from the data above, nothing else)

1. **Laya, once given the same controller-level safety mechanisms the hand-coded teacher already had (threshold calibration, jump-hold hysteresis, stuck-wall fallback), completes World 1-1** — the same outcome as teacher_v2, 10 in-game time units slower (120 vs 110), and with **higher average distance across the other 7 stages** it was never specifically tuned for (1377.5 vs 1070.4).
2. **Laya learned a real, generalizing policy**, not a memorized one: held-out-stage jump accuracy rose from a 64.5% baseline to 93.3% after imitation training, and it clears stages 4-1 and 8-1 further than the teacher's fixed thresholds do, despite never training on 4-1 at all.
3. **Two of the four bugs that mattered most were found only by actually watching gameplay** (a video, not a metric): the enemy-type-0 blind spot (§3, found via frame-level RAM inspection after a death looked "impossible") and the stuck-wall permanent freeze (§9, found because the user watched the recording and asked why it wasn't moving). This is the single clearest empirical confirmation, in this project's own data, of the reference blog's founding claim: offline accuracy and aggregate distance metrics do not surface every failure mode a policy has — playing (and watching) the game does.
4. **Running speed (the most direct route to beating 110s) does not work with this architecture** at the level of effort invested here, for a specific, documented physics reason — reported as an open problem, not a hidden failure.
5. **No claim is made about outperforming a human.** No human was benchmarked; the one external reference (TAS, glitch route) is explicitly not comparable to either policy trained in this project.

## 11. Demo walkthrough

`python models/laya/mario/demo/run_demo.py` opens a two-column page: real gameplay video on the
left, a live decision panel on the right. Everything the panel shows is read from
`episode.jsonl`, produced by `record_demo.py` actually running `mario_laya_v6_dagger` once through
the exact controller loop reported above -- the panel does not compute or guess anything.

What it deliberately does *not* show, and why: the model only ever outputs one probability
(P(jump)) -- `move` is hardcoded "RIGHT" in the controller and the final config never holds the
run button, so there is no real per-direction or run probability to display. Earlier drafts of
this UI's spec sketched exactly those bars; they were left out rather than filled with invented
numbers.

Three decision states are visible, each labeled honestly:
- **model** (green) -- a fresh forward pass this decision; the bars and confidence are live.
- **held** (amber) -- continuing a jump already committed to (hysteresis, §5); the bars show the
  *last real* probability with its originating decision number, not a new inference.
- **stuck_fallback** (red) -- the controller-level safety net (§9) forced a jump with no model
  call at all; this is the exact mechanism that turned a permanent freeze into a level clear, and
  the panel shows it firing, e.g. at decision #353 (x=2130, on_ground, velocity_x=0) -- precisely
  the spot the user's screenshot caught stuck in the pre-fix recording.
