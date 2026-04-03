# SASNet Data Adaptation Plan

Date: 2026-04-04

## Objective

Map the current project datasets `GWHD_2021`, `MTC`, and `URC` into the data contract required by `SASNet`.

## Upstream Data Contract

From the vendored repository:

- `main.py` uses `datasets/loading_data.py`
- `datasets/crowd_dataset.py` reads:
  - images from `test_data/images/*.jpg`
  - density maps from `ground_truth/*_sigma4.h5`

So the effective upstream inference contract is:

- sample: `(image_tensor, density_map)`

The official snapshot does not expose a separate training dataset class, so local adaptation must define one.

## Local Dataset Mapping

### GWHD_2021

- Source protocol in project: boxes converted to point supervision
- Local adaptation target:
  - generate density maps from points
  - expose train tuples `(image, density_map)`
  - expose val tuples `(image, count, name)`

### MTC

- Source protocol in project: point-counting dataset with split files
- Local adaptation target:
  - reuse project split logic
  - generate density maps from points
  - expose train and val tuples through a wrapper

### URC

- Source protocol in project: point-counting dataset at larger input resolution
- Local adaptation target:
  - keep project split semantics
  - generate density maps from points
  - expose wrapper-based train and val tuples

## Preferred Adapter Design

Implement local wrappers instead of mutating upstream dataset code:

```text
external/baselines/sasnet/local_adapters/
  datasets.py
  density_targets.py
```

Where:

- `datasets.py` builds train/val adapters from current project configs
- `density_targets.py` handles `sigma=4`-style density target generation or nearest compatible local equivalent

## Key Compatibility Questions

Before implementation, confirm:

1. whether local density targets should match upstream `sigma4` exactly
2. whether upstream model expects any implicit `log_para` scaling during training
3. whether train-time augmentation should follow project transforms or a SASNet-specific policy

## Recommendation

Do not generate `_sigma4.h5` files on disk unless wrapper-based density generation becomes too brittle. Prefer in-memory target generation from the current project pipeline.

## Deliverable

After implementation, the adapter should be able to build:

- train dataloader for `GWHD_2021`
- val dataloader for `GWHD_2021`
- same pattern for `MTC`
- same pattern for `URC`
