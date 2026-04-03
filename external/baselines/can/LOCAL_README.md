# CAN Local Intake README

This document is the local source of truth for how `CAN` is pinned and staged inside this repository.

## Upstream Identity

- Upstream URL: `https://github.com/weizheliu/Context-Aware-Crowd-Counting`
- Upstream branch: `master`
- Pinned commit: `d2e4d0425f578e556c1ab6017d326cff20466fad`
- Pin date: `2026-04-04`
- Storage model: vendored snapshot

## Local Directory Contract

```text
external/baselines/can/
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
- deciding whether `CAN` should proceed to local adaptation planning

## Non-Goals

This intake slice does not:

- modify `pack/train.py`
- register `CAN` inside `pack.models`
- add dataset wrappers for `GWHD_2021`, `MTC`, or `URC`
- add unified metric-export scripts
- add complexity wrappers for the upstream model

## Required Audit Targets After Vendoring

- `train.py`
- `test.py`
- `dataset.py`
- `image.py`
- `model.py`
- `README.md`

## First-Pass Audit

- Training entry: `train.py`
- Validation and test entry: `test.py`
- Upstream dataset protocol:
  - train CLI takes `train.json` and `val.json`
  - dataset rows are image paths
  - labels are loaded from `.h5` density maps next to images via `image.py`
- Train batch contract from `dataset.py`:
  - `(image, density_map)`
- Validation path in `test.py` computes count by summing predicted density maps and compares against `.h5` density sums
- `image.py` applies:
  - quarter selection style random crop at half image size
  - optional horizontal flip
  - density downsampling by `8x` with scale compensation `*64`
- Model definition is isolated in `model.py` as `CANNet`
- Complexity measurement should be feasible by instantiating `CANNet`, but current code will try to download VGG16 weights unless patched or monkey-patched
- README declares:
  - `PyTorch 0.4.1`
  - `Python 2.7`
- Code quality risks are real but bounded:
  - uses `xrange`
  - uses old `Variable` patterns
  - uses integer division assumptions such as `/`
  - test script hard-codes paths
- Structural readability is good enough to continue into local adaptation planning

## Current Status

- Upstream pin: complete
- Vendor snapshot: present under `external/baselines/can/upstream/`
- First-pass audit: captured
- Local adaptation planning: approved to continue
- Unified metric export: pending
- Complexity integration: pending

## Next Files To Create

- `external/baselines/can/LOCAL_ADAPTATION_NOTES.md`
- `external/baselines/can/DATA_ADAPTATION_PLAN.md`
- `external/baselines/can/RESULT_EXPORT_PLAN.md`
