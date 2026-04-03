# SASNet Local Adaptation Notes

Date: 2026-04-04

## Purpose

This document records the local interpretation of how the official `SASNet` repository should be adapted to the current project protocol without rewriting the method into a different baseline.

## What Must Stay True To Upstream

- Keep `SASNet` as the core model definition.
- Preserve the original density-regression output style.
- Preserve count evaluation by summing predicted density maps.
- Preserve the scale-selection structure defined in `model.py`.

## What Needs Local Adaptation

- Replace the upstream fixed `test_data/images` directory contract with project-aware dataset wrappers for `GWHD_2021`, `MTC`, and `URC`.
- Replace `_sigma4.h5` file assumptions with project-side density target generation or wrapper-based target construction.
- Replace hard-coded inference entry behavior with reusable evaluation helpers that emit `MAE`, `MSE`, and `MAPE`.
- Add a local complexity wrapper that can instantiate `SASNet` without uncontrolled pretrained-weight download.
- Add local train/eval bridge code because the official snapshot does not expose a standalone training script.

## Upstream Constraints

- README declares `python 3.6.8` and `pytorch >= 1.5.0`.
- The official repository snapshot is inference-oriented and does not expose a dedicated training entry.
- `main.py` assumes pre-generated density maps and pre-trained model weights.
- `prepare_dataset.py` assumes `.txt` point annotations and writes `_sigma4.h5` density maps.
- `model.py` depends on VGG16-BN features and legacy `upsample_*` APIs.

## Local Compatibility Strategy

The preferred local adaptation route is:

1. keep upstream code vendored and untouched under `external/baselines/sasnet/upstream/`
2. add project-owned bridge code under `external/baselines/sasnet/local_adapters/`
3. implement train/eval/complexity wrappers outside the vendored code first
4. only patch upstream files directly if wrapper-based adaptation becomes impossible

## Result Type

Unless the official code runs with zero structural adaptation under the current protocol, `SASNet` should be reported as `adapted reproduction`.

## Next Artifacts

- `DATA_ADAPTATION_PLAN.md`
- `RESULT_EXPORT_PLAN.md`
