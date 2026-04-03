# DM-Count Result Export Plan

Date: 2026-04-04

## Goal

让 `DM-Count` 在保留 upstream 训练逻辑的同时，输出符合当前论文协议的统一结果字段与复杂度统计。

## Required Unified Metrics

Paper-facing metrics must be:

- `MAE`
- `MSE`
- `MAPE`

Current upstream behavior:

- reports `MAE`
- reports `MSE`
- does not report `MAPE`

## Metric Export Strategy

Recommended local files:

- `external/baselines/dm_count/local_adapters/eval.py`
- `external/baselines/dm_count/local_adapters/export_results.py`

### Eval Wrapper Responsibilities

- run the upstream model on local validation/test data
- collect predicted counts and ground-truth counts
- compute:
  - `MAE`
  - `MSE`
  - `MAPE`
- emit a row format aligned with current paper tables

### Export Responsibilities

- save per-run summary as `.json` or `.csv`
- include:
  - method name
  - dataset
  - split
  - checkpoint path
  - mae
  - mse
  - mape
  - reproduction label

## Reproduction Label

For this method the default paper label should be:

- `adapted reproduction`

Only upgrade to `local reproduction` if the final path keeps the official method semantics intact and only changes data/input plumbing.

## Complexity Wrapper Strategy

Recommended local file:

- `external/baselines/dm_count/local_adapters/measure_complexity.py`

Purpose:

- instantiate upstream `vgg19()` model
- switch to `eval()`
- forward a dummy tensor with shape `1 x 3 x 1080 x 1920`
- compute local `Params` and `Approx FLOPs`

## Complexity Measurement Contract

Use the same protocol as current local baselines:

- input shape: `1 x 3 x 1080 x 1920`
- single forward
- no TTA
- `float32`
- report:
  - `Params`
  - `Approx FLOPs`

## Implementation Warning

The current `pack/tools/measure_model_complexity.py` assumes a `pack.models.build_model()` path.

For `DM-Count`, do not force that tool to know about upstream repositories. Instead:

- either import and reuse its internal counting logic
- or duplicate only the minimal FLOP-count wrapper in an external adapter script

This keeps `pack/` and `external/` boundaries clean.

## Final Table Fields

The minimum result row for paper use should include:

- `Method`
- `Dataset`
- `Params`
- `Approx FLOPs`
- `MAE`
- `MSE`
- `MAPE`
- `Result Type`

Where `Result Type` is one of:

- `local reproduction`
- `adapted reproduction`
- `literature citation`

## Success Criteria

Result export and complexity integration are complete when:

- at least one `DM-Count` run can export `MAE / MSE / MAPE`
- one local complexity report can be generated for upstream `vgg19()`
- the exported fields can be copied directly into the current manuscript comparison tables
