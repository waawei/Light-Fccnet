# CAN Data Adaptation Plan

Date: 2026-04-04

## Objective

Map the current project datasets `GWHD_2021`, `MTC`, and `URC` into the data contract expected by the official `CAN` training and evaluation logic.

## Upstream Data Contract

From the vendored repository:

- `train.py` expects:
  - `train.json`
  - `val.json`
- `dataset.py` expects each JSON item to be an image path
- `image.py` converts each image path into:
  - RGB image
  - `.h5` density map loaded by replacing `images` with `ground_truth`

So the effective upstream training contract is:

- train sample: `(image_tensor, density_map)`
- eval sample: `(image_tensor, density_map)`

## Local Dataset Mapping

### GWHD_2021

- Source protocol in project: box annotations converted to point supervision
- Local adaptation target:
  - generate density maps from points using project-side density utilities
  - expose image-plus-density pairs to a `CAN` adapter dataset

### MTC

- Source protocol in project: point-counting dataset with split files
- Local adaptation target:
  - reuse project dataset split logic
  - generate density maps from point annotations
  - expose image-plus-density pairs to the `CAN` adapter dataset

### URC

- Source protocol in project: point-counting dataset under larger input resolution
- Local adaptation target:
  - keep project split semantics
  - generate density maps from points
  - expose image-plus-density pairs to the `CAN` adapter dataset

## Preferred Adapter Design

Implement a local wrapper instead of mutating upstream dataset code:

```text
external/baselines/can/local_adapters/
  datasets.py
  density_targets.py
```

Where:

- `datasets.py` builds train/val adapters from current project configs
- `density_targets.py` handles target generation and any `8x` downsampling alignment needed by `CAN`

## Key Compatibility Questions

Before implementation, confirm:

1. whether the current project density generation exactly matches the scale assumptions expected by `CAN`
2. whether `image.py` style random crop logic should be replicated in the adapter or replaced with project transforms
3. whether `CAN` should keep upstream half-size quadrant evaluation or use a project-standard evaluation wrapper

## Recommendation

Do not generate `.h5` files on disk just to mimic upstream layout unless wrapper-based adaptation becomes too brittle. Prefer in-memory density targets from the current project pipeline.

## Deliverable

After implementation, the adapter should be able to build:

- train dataloader for `GWHD_2021`
- val dataloader for `GWHD_2021`
- same pattern for `MTC`
- same pattern for `URC`
