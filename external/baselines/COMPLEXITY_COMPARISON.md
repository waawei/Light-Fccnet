# Baseline Complexity Comparison

Date: 2026-04-04

## Purpose

This table consolidates the currently available local complexity measurements for the horizontal baselines that have already been implemented or bridged.

## Measurement Protocol

- Input shape: `1 x 3 x 1080 x 1920`
- Local measurement utilities:
  - `pack/tools/measure_model_complexity.py`
  - baseline-specific wrappers under `external/baselines/*/local_adapters/measure_complexity.py`
- FLOPs are approximate hook-based counts and should be reported as local reproduction values, not official paper numbers

## Current Table

| Method | Params | Approx FLOPs | Params (human) | FLOPs (human) | Status | Measurement Source |
| --- | ---: | ---: | --- | --- | --- | --- |
| Light-FCCNet | 915,860 | 2,495,492,928,948 | 0.92M | 2495.49G | merged on `main` | `pack/config/gwhd/config_gwhd_light_full.yaml` |
| CSRNet | 16,263,489 | 1,713,440,239,200 | 16.26M | 1713.44G | merged on `main` | `pack/config/gwhd/config_gwhd_csrnet.yaml` |
| DM-Count | 21,499,457 | 1,706,845,012,320 | 21.50M | 1706.85G | merged on `main` | `external/baselines/dm_count/local_adapters/measure_complexity.py` |
| CAN | 18,103,489 | 1,815,533,380,400 | 18.10M | 1815.53G | measured in `can-adapters` worktree | `external/baselines/can/local_adapters/measure_complexity.py` |
| SASNet | 38,898,698 | 3,675,741,101,280 | 38.90M | 3675.74G | merged on `main` | `external/baselines/sasnet/local_adapters/measure_complexity.py` |

## Interpretation Notes

- `Light-FCCNet` remains the most parameter-efficient model among the currently bridged baselines.
- `CSRNet` and `DM-Count` have similar approximate FLOPs under this protocol, but `DM-Count` carries a much larger parameter count than `Light-FCCNet`.
- `CAN` is not yet merged back to `main`, so its complexity row is currently sourced from the validated `can-adapters` worktree.
- `SASNet` is the heaviest model in the current comparison set by both parameters and FLOPs.

## Scope Boundary

This table intentionally excludes:

- `CMTL`
- `M-SegNet`

Reason:

- they are not part of the current local reproduction set
- no unified local complexity wrapper has been accepted into the project for them

## Paper Usage Guidance

For the manuscript:

- this table can be used as the internal source of truth for local complexity values
- if the final paper table uses rounded values, preserve the raw values above in repo docs
- if `CAN` is later merged into `main`, update only the `Status` column; the measured values should remain the same unless the wrapper changes
