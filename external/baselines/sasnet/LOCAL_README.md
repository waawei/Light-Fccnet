# SASNet Local Intake README

This document is the local source of truth for how `SASNet` is pinned and staged inside this repository.

## Upstream Identity

- Upstream URL: `https://github.com/TencentYoutuResearch/CrowdCounting-SASNet`
- Upstream branch: `main`
- Pinned commit: `3e2b78a6c6ebe761c5be6a9181457daad6df666d`
- Pin date: `2026-04-04`
- Storage model: vendored snapshot

## Local Directory Contract

```text
external/baselines/sasnet/
  LOCAL_README.md
  upstream/
```

- `LOCAL_README.md` stores local integration intent, pin metadata, and first-pass audit notes.
- `upstream/` stores a pinned copy of the official repository working tree.
- `upstream/.git` must not exist inside this repository.

## Current Scope

This intake slice only covers:

- pinning the official upstream repository as a vendored snapshot
- recording first-pass audit facts needed for the next adaptation slice
- deciding whether `SASNet` should proceed to local adaptation planning

## Non-Goals

This intake slice does not:

- modify `pack/train.py`
- register `SASNet` inside `pack.models`
- add dataset wrappers for `GWHD_2021`, `MTC`, or `URC`
- add unified metric-export scripts
- add complexity wrappers for the upstream model

## Required Audit Targets After Vendoring

- `main.py`
- `model.py`
- `prepare_dataset.py`
- `datasets/`
- `requirements.txt`
- `README.md`

## First-Pass Audit

- Main upstream entry: `main.py`
- Upstream currently exposes inference only; no dedicated training script is present in the official repository snapshot
- Data loading entry: `datasets/loading_data.py`
- Dataset contract from `datasets/crowd_dataset.py`:
  - test-time dataset rooted at `test_data/images/*.jpg`
  - labels loaded from `_sigma4.h5` density maps under `ground_truth`
  - sample tuple: `(image, density_map)`
- Density generation entry: `prepare_dataset.py`
  - consumes `.txt` point annotations
  - writes `_sigma4.h5` density maps
- Model definition is isolated in `model.py` as `SASNet`
- `main.py` reports `mae` and `mse` from count-by-summing density maps
- `MAPE` is not exposed upstream
- Complexity measurement should be feasible by instantiating `SASNet`, but model construction will try to load VGG16-BN pretrained weights unless patched or monkey-patched
- README declares:
  - `python 3.6.8`
  - `pytorch >= 1.5.0`
- `requirements.txt` pins an older but still Python-3-era stack:
  - `h5py==3.1.0`
  - `numpy==1.19.0`
  - `opencv-python==4.4.0.46`
  - `Pillow==8.0.1`
  - `scipy==1.5.4`
  - `matplotlib==3.3.3`
- Structural readability is good enough to continue into local adaptation planning
- Major functional risk:
  - official snapshot appears inference-oriented and may require local bridge code for training under current project protocol

## Current Status

- Upstream pin: complete
- Vendor snapshot: present under `external/baselines/sasnet/upstream/`
- First-pass audit: captured
- Local adaptation planning: approved to continue
- Unified metric export: pending
- Complexity integration: pending

## Next Files To Create

- `external/baselines/sasnet/LOCAL_ADAPTATION_NOTES.md`
- `external/baselines/sasnet/DATA_ADAPTATION_PLAN.md`
- `external/baselines/sasnet/RESULT_EXPORT_PLAN.md`
