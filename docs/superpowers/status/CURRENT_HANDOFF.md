# Current Handoff

Date: 2026-04-04

## Current Goal

`SASNet` local adapter layer and local bridge layer are complete in this worktree. The next step is a local train/eval smoke and then `autodl` execution guidance.

## Repository Root

`d:\develop\python\SecondChoice`

## Worktree Context

- Worktree: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters`
- Branch: `sasnet-adapters`
- Main code directory remains `pack/`
- Do not reintroduce `code/` path usage

## Completed In This Worktree

- Added plan:
  - `docs/superpowers/plans/2026-04-04-sasnet-adapter-implementation.md`
- Added adapter files:
  - `external/baselines/sasnet/local_adapters/__init__.py`
  - `external/baselines/sasnet/local_adapters/density_targets.py`
  - `external/baselines/sasnet/local_adapters/datasets.py`
  - `external/baselines/sasnet/local_adapters/eval.py`
  - `external/baselines/sasnet/local_adapters/export_results.py`
  - `external/baselines/sasnet/local_adapters/measure_complexity.py`
- Added bridge files:
  - `external/baselines/sasnet/local_adapters/runner.py`
  - `external/baselines/sasnet/local_adapters/train_bridge.py`
  - `external/baselines/sasnet/local_adapters/run_local.py`
- Added tests:
  - `tests/test_sasnet_adapters.py`

## Fresh Verification Evidence

Executed in this worktree:

```powershell
python -m unittest tests.test_sasnet_adapters -v
python -m unittest tests.test_sasnet_adapters tests.test_dm_count_adapters tests.test_baseline_complexity -v
python -c "from external.baselines.sasnet.local_adapters.measure_complexity import measure_sasnet_complexity; print(measure_sasnet_complexity((1,3,1080,1920)))"
```

Results:

- `tests.test_sasnet_adapters`: 18 tests passed
- Combined regression suite: 40 tests passed
- Complexity result:
  - `SASNet`: `38,898,698` params
  - `3675.74G` approx FLOPs
  - input shape: `(1, 3, 1080, 1920)`

Warnings observed but not blocking:

- `torchvision` warns that upstream still uses deprecated `pretrained`
- upstream `model.py` uses deprecated `upsample_bilinear` and `upsample_nearest`
- `albumentations` warns about keypoint processor setup on one dataset-builder path

## Upstream Pin

- upstream URL: `https://github.com/TencentYoutuResearch/CrowdCounting-SASNet`
- upstream branch: `main`
- pinned commit: `3e2b78a6c6ebe761c5be6a9181457daad6df666d`
- vendored snapshot path: `external/baselines/sasnet/upstream/`

## Non-Negotiable Context

- Formal paper metrics `MAE / MSE / MAPE` must come from full runs on `autodl`
- Local results here are only for adapter validation, export compatibility, and complexity measurement
- Do not modify `pack/train.py` for SASNet
- Do not register `SASNet` in `pack.models` during adapter-only work

## Next Recommended Step

1. Run one local `run_local.py` smoke with a real config and minimal epoch count
2. Prepare `autodl` commands for `GWHD / MTC / URC`
3. Merge this worktree slice back once smoke is clean
