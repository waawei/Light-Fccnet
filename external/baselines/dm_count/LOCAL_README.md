# DM-Count Local Intake README

This document is the local source of truth for how `DM-Count` is pinned and staged inside this repository.

## Upstream Identity

- Upstream URL: `https://github.com/cvlab-stonybrook/DM-Count`
- Upstream branch: `master`
- Pinned commit: `cc5f2132e0d1328909f31b6d665b8e0b15c30467`
- Pin date: `2026-04-04`
- Storage model: vendored snapshot

## Local Directory Contract

```text
external/baselines/dm_count/
  LOCAL_README.md
  upstream/
```

- `LOCAL_README.md` stores local integration intent, pin metadata, and first-pass audit notes.
- `upstream/` stores a pinned copy of the official repository working tree.
- `upstream/.git` must not exist inside this repository.

## Current Scope

This intake slice only covers:

- correcting immediate DM-Count integration documents to `pack/`-aligned paths
- pinning the official upstream repository as a vendored snapshot
- recording first-pass audit facts needed for the next adaptation slice

## Non-Goals

This intake slice does not:

- modify `pack/train.py`
- register `DM-Count` inside `pack.models`
- add dataset wrappers for `GWHD_2021`, `MTC`, or `URC`
- add unified metric-export scripts
- add complexity wrappers for the upstream model

## Required Audit Targets After Vendoring

- `train.py`
- `train_helper.py`
- `test.py`
- `datasets/crowd.py`
- `requirements.txt`

## First-Pass Audit

- Training entry: `train.py`
- Validation and test entry: `test.py`
- Upstream datasets exposed by CLI: `qnrf`, `nwpu`, `sha`, `shb`
- `train.py` applies hard-coded crop-size overrides by dataset:
  - `qnrf`: `512`
  - `nwpu`: `384`
  - `sha`: `256`
  - `shb`: `512`
- `train_helper.py` expects:
  - train batch: `(image, points, gt_discrete)`
  - val batch: `(image, count, name)`
- `datasets/crowd.py` consumes:
  - `.jpg` plus `.npy` point files for QNRF and NWPU
  - `.mat` annotations for ShanghaiTech
- `test.py` reports `mae` and `mse`, but does not expose `MAPE`
- `requirements.txt` pins an old stack including `torch==1.2.0` and `torchvision==0.4.0`
- Compatibility repair is expected later before any real local adaptation run

## Current Status

- Upstream pin: complete
- Vendor snapshot: present under `external/baselines/dm_count/upstream/`
- First-pass audit: captured
- Data adaptation: pending
- Unified metric export: pending
- Complexity integration: pending

## Next Files To Create

- `external/baselines/dm_count/LOCAL_ADAPTATION_NOTES.md`
- `external/baselines/dm_count/DATA_ADAPTATION_PLAN.md`
- `external/baselines/dm_count/RESULT_EXPORT_PLAN.md`
