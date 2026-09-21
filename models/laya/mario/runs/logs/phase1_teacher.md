# Phase 1: Teacher strengthening -- results log

**Date:** 2026-09-21
**Env:** SuperMarioBros-1-1-v0 (gym_super_mario_bros 9.1.0, nes-py 9.0.1, gymnasium 1.3.0)
**Hardware:** RTX 5070 Ti, torch 2.11.0+cu128, transformers 5.17.0
**Determinism note:** this environment has no source of randomness in the tested policies (no
sticky actions, no env seed variance) -- repeated runs of the same deterministic policy produce
byte-identical trajectories. Single-run distance is therefore a valid, reproducible metric here,
but note it for data collection: a single deterministic rollout gives near-zero state diversity,
so data collection must inject variation (see Phase 3 plan).

## RAM addresses (empirically validated, not from memory alone)

Discovered/confirmed in `experiments/ram_probe.py` and `experiments/tile_probe.py`:

| Field | Address | Validation method |
|---|---|---|
| Player page / screen-x | 0x6D / 0x86 | Already used by gym_super_mario_bros itself (verified in its source) |
| Enemy page / screen-x (per slot) | 0x6E+slot / 0x87+slot | Structurally adjacent to player's own verified addresses; value trajectory matches frame-diff pixel tracking (mean abs error ~4px over a 78-frame smooth chain) |
| Enemy screen-y (per slot) | 0xCF+slot, +16px offset | 184 + 16 == 200, matching frame-diff ground truth almost exactly (this is the key confirmation -- a naive value-correlation search over all 2048 RAM bytes was confounded by many incidentally-constant bytes and did NOT find this address on its own) |
| Enemy drawn/active flag | 0x0F+slot | Cross-checked against enemy_types (0x16+slot, already verified in the installed package source) being simultaneously nonzero |
| Tile grid (16x13, ground/gap) | 0x500 + page*0xD0 + sub_y*16 + sub_x | Validated by construction of `ground_ahead()`: correctly identifies solid pipes (blocks horizontal movement) vs actual gaps; used successfully by RuleTeacherV2 to jump-clear multiple real pits |

## Policy comparison (single deterministic rollout each, World 1-1)

| Policy | Distance | Outcome | Notes |
|---|---|---|---|
| Random | ~untested this round | -- | (see Phase 0 mario_agent.py for random policy scaffold) |
| Laya zero-shot (typed-decisions checkpoint, no fine-tuning) | 0 | Stuck at spawn | Constant ~62% P(left) regardless of state; confirms README's own "near chance zero-shot outside training domain" limitation |
| RuleTeacherV1 (info-dict only: enemy count, on_ground, periodic+reactive jump) | 722 | Blocked by pipe (no wall-stuck detection in first version) / later: dies to enemy (after jump-hold fix) | No RAM; only sees the same crude fields the naive Laya state extractor exposed |
| **RuleTeacherV2 (RAM: real enemy dx/dy, tile-based gap detection, stuck-against-wall detection)** | **2754** | Dies to enemy during a multi-gap jump sequence near the end of the level | Level length ~3160-3260; this clears ~85-87% of World 1-1, including every pipe and most pits, using only reactive rules (no lookahead planning) |

**Best RuleTeacherV2 config:** `enemy_jump_dx=28, gap_lookahead_px=24, hold_decisions=5, stuck_threshold=3`
(swept 8 configs total; hold_decisions=4 regresses hard to 722 -- confirms jump-hold duration, not
reaction thresholds, was the binding constraint for clearing pipes).

## Decision for Phase 2/3

RuleTeacherV2 is the "strong teacher" whose rollouts will be imitated. It is not perfect (dies
before the flag), which is expected and consistent with the reference blog post (Doom/Pong
fitted rules also didn't discover new strategies, just recovered the teacher's own competence
ceiling) -- the fine-tuned Laya's job is to match this teacher, not exceed it, in the first pass.
DAgger (Phase 7) is where a learner could eventually specialize beyond imitation.
