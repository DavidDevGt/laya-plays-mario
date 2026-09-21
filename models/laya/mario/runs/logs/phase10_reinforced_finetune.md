# Phase 10: "reinforce with more successful trajectories" -- tried, did not beat v6

## Hypothesis

`mario_laya_v6_dagger`'s training data contained very few examples of *completing* World 1-1 (one
clean epsilon=0 run, one lucky epsilon=0.08 run) relative to episodes that die early on other
stages. Hypothesis: Laya's borderline probabilities (0.39-0.43 against a 0.5 threshold) at the
safety-critical frames near the end of the level came from insufficient repeated signal for
"this exact situation -> jump," and collecting many more slightly-varied *completing* trajectories
on 1-1 specifically would sharpen that signal.

## What was done

Collected 25 additional episodes on World 1-1 alone (epsilon=0.05, seed=10): **8/25 completed the
level** (flag_get=True), each with slightly different intermediate states due to exploration noise
-- exactly the kind of repeated-but-varied "this is correct here" signal the hypothesis called for.
Combined with the existing `v6_aggregated.jsonl` (12090 decisions) into `v7_final.jsonl` (21887
decisions, 36% jump rate) and retrained fresh from the `typed-decisions` base (not continued from
v5/v6), 15 epochs, seed 0 -> `mario_laya_v7_speedrunner`. Held-out (4-1, 7-1) jump accuracy: 85.0%
(vs v6's 82.9%, v5's 93.3%).

## Result: worse gameplay despite comparable-or-better offline accuracy

Swept threshold x hold_decisions x stuck_threshold extensively (>25 configs) -- the same search
that found v6's clearing configuration in a handful of tries. **Best found for v7: distance 2754
(threshold=0.65, hold_decisions=8, stuck_threshold=8) -- does not complete the level.** No
configuration tried reached World 1-1's flag, versus v6's clean, easily-found clear.

This is reported as a genuine negative result, not smoothed over: **more successful-trajectory
data and better held-out accuracy did not transfer to better gameplay.** Plausible explanations
(not verified further, given time budget):
- Overrepresenting near-duplicate late-level states (from 8 similar completing runs on one stage)
  may have shifted the decision boundary in a way that helps those specific states but hurts
  calibration on the wider mix of states the controller actually needs to navigate.
- 21887 decisions on a 26.5M-parameter head is not a large-data regime; adding ~1.7x more data
  concentrated on one stage's late-game states is a distribution shift, not just more signal.
- The controller-level hysteresis parameters (hold/stuck) that were tuned for v6 do not
  necessarily transfer to v7's different probability landscape, and the reported sweep, while
  broad, is not exhaustive -- a config that clears the level for v7 may exist outside the region
  searched.

## Decision

**`mario_laya_v6_dagger` (threshold=0.5, hold_decisions=4, stuck_threshold=8) remains the final,
best-performing checkpoint** -- it clears World 1-1 in 120 in-game time units, matching
`teacher_v2`'s completion. `mario_laya_v7_speedrunner` is kept (not deleted) as a documented,
unsuccessful attempt, consistent with the project's standing instruction not to hide failed
hypotheses.

## Lesson

This is the same lesson the project has surfaced repeatedly (phases 4/5, 7, 9), from yet another
angle: **an intuitively sound data-augmentation idea, backed by improved offline metrics, still
has to be verified by actually playing the game.** It was not assumed to work because the
reasoning sounded right; it was tested, found wanting, and reported as such.
