# CAN Local Adaptation Notes

Date: 2026-04-04

## Purpose

This document records the local interpretation of how the official `CAN` repository should be adapted to the current project protocol without rewriting the method into a different baseline.

## What Must Stay True To Upstream

- Keep `CANNet` as the core model definition.
- Preserve the original density-regression training target style.
- Preserve the count evaluation logic based on density-map summation.
- Preserve the upstream assumption that the network outputs a single-channel density map.

## What Needs Local Adaptation

- Replace upstream path-list JSON handling with project-aware dataset wrappers for `GWHD_2021`, `MTC`, and `URC`.
- Replace upstream `.h5` density-map loading assumptions with project-side density target generation or reuse of existing local density supervision.
- Replace hard-coded evaluation path logic with reusable evaluation helpers that emit `MAE`, `MSE`, and `MAPE`.
- Add a local complexity wrapper that can instantiate `CANNet` without uncontrolled pretrained-weight downloads.

## Upstream Constraints

- README declares `Python 2.7` and `PyTorch 0.4.1`.
- `model.py` uses `xrange` and old torchvision VGG initialization assumptions.
- `train.py` uses old `Variable` patterns and legacy loss argument style.
- `test.py` hard-codes image folders and checkpoint paths.
- `image.py` assumes density maps are stored as `.h5` files parallel to images.

## Local Compatibility Strategy

The preferred local adaptation route is:

1. keep upstream code vendored and untouched under `external/baselines/can/upstream/`
2. add project-owned bridge code under `external/baselines/can/local_adapters/`
3. perform compatibility repair in adapters or wrapper entry points first
4. only patch upstream code directly if wrapper-based adaptation becomes impossible

## Result Type

Unless the official code runs with zero structural adaptation under the current protocol, `CAN` should be reported as `adapted reproduction`.

## Next Artifacts

- `DATA_ADAPTATION_PLAN.md`
- `RESULT_EXPORT_PLAN.md`
