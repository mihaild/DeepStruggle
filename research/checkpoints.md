# Checkpoint catalogue — what exists on disk, and what it is worth

One row per rated checkpoint, with its measured strength and the field that measured it. The arms
themselves are [`runs.md`](runs.md); this file answers the narrower "which model do I load, and how
good is it".

**Ladder reset 2026-09-19.** Pre-P17 checkpoints cannot be loaded on this engine at all — their
policy head is 212 wide against the current 220, and `check_checkpoint_layout` refuses them by
width rather than letting them misread. They are in
`/workspace/data/archive/E3_ladder/checkpoints/` with their ratings in
[`archive/E3_ladder/checkpoints.md`](archive/E3_ladder/checkpoints.md), and none of those ratings
transfers.

## How to read a number here

A win rate is only meaningful with its opponent, its game count and its side split. Until E4 has a
frozen anchor, the only cross-engine-comparable opponents are the rule-based bots, so every row
below is against `HeuristicBot` at 50 games a side.

## Live checkpoints

| checkpoint | steps | vs HeuristicBot | US / USSR | note |
|:---|---:|---:|:---|:---|
| `E4_1_warmup.pt` | — | 35% | 26 / 44 | BC warmup, 2 epochs on the rebuilt human corpus (254 games, 111,203 samples); 38.65% agreement |
| `E4-02-01 .../snapshot_*` | to 240M | 93–98% | balanced within ~4pp | in flight; see `runs.md` |
| `E4-01-01 .../snapshot_*` | to 184M | peaked 95% @105M | collapsed to 80 / 98 | **aborted, misconfigured** — do not rate |

## Rebuilding a rating

The engine changed, so anything that consumed the old decision stream is void: `(seed, actions)`
datasets truncate silently, and Elo anchors predate the change. `tools/tournament.py` over a run
directory is the way to produce a fresh ladder once E4-02-01 finishes; the human corpus dataset was
already rebuilt against this engine as `/workspace/data/datasets/human_corpus_e4`.
