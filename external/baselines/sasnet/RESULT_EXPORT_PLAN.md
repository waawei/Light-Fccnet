# SASNet Result Export Plan

Date: 2026-04-04

## Objective

Ensure `SASNet` results can be exported under the same paper-facing protocol used by the rest of the project.

## Required Metrics

The local project comparison protocol requires:

- `MAE`
- `MSE`
- `MAPE`

Upstream `SASNet` currently reports:

- `MAE`
- `MSE`

So local adaptation must add:

- explicit `MAPE`
- project-standard naming and row formatting

## Export Contract

The paper-facing result row should include at least:

- `method`
- `dataset`
- `params`
- `flops`
- `mae`
- `mse`
- `mape`
- `result_type`
- `notes`

## Preferred Export Design

Use project-owned helpers instead of modifying upstream `main.py` directly:

```text
external/baselines/sasnet/local_adapters/
  eval.py
  export_results.py
  measure_complexity.py
```

Where:

- `eval.py` computes project-standard metrics from prediction and ground-truth counts
- `export_results.py` writes JSON and CSV summary rows
- `measure_complexity.py` instantiates `SASNet` in a controlled way for `1 x 3 x 1080 x 1920`

## Complexity Policy

Complexity must be measured locally under the unified project rule:

```powershell
python pack\tools\measure_model_complexity.py --config <config> --input-shape 1 3 1080 1920
```

If `SASNet` cannot be measured cleanly through the shared tool, add a local wrapper and clearly mark it as an approximate aligned measurement.

## Output Classification

Until fully validated under the current project protocol, `SASNet` results should be labeled:

- `adapted reproduction`

## Deliverable

After implementation, `SASNet` should be able to produce:

- local complexity report
- JSON summary row
- CSV-appendable paper result row
- metrics aligned to `MAE / MSE / MAPE`
