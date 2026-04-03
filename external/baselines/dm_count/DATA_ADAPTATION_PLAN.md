# DM-Count Data Adaptation Plan

Date: 2026-04-04

## Goal

将 `GWHD_2021`、`MTC`、`URC` 映射到 `DM-Count` 官方训练语义，而不直接重写其核心训练逻辑。

## Adaptation Principle

优先适配到 upstream `DM-Count` 的数据接口，而不是强迫 upstream 迁就 `pack/` 主训练循环。

## Upstream Expected Shapes

### Train

Expected tuple:

- `image`
- `points`
- `gt_discrete`

### Val/Test

Expected tuple:

- `image`
- `count`
- `name`

## Local Dataset Mapping

### GWHD_2021

Current local truth:

- source labels are wheat-head boxes
- local loader converts boxes to center points

DM-Count mapping:

- keep current box-to-point conversion
- preserve current train/val split files already used by `pack`
- for training, generate:
  - transformed image tensor
  - variable-length point tensor
  - downsampled discrete map
- for validation, output:
  - transformed image tensor
  - integer count
  - image name

Risk:

- `GWHD` does not naturally ship as `.npy` point files, so direct upstream dataset classes cannot be reused unchanged

### MTC

Current local truth:

- source labels may come from `.mat` or UAV-style CSV
- local loader already normalizes everything into point coordinates

DM-Count mapping:

- reuse current split files
- standardize to point arrays before entering upstream train loop
- keep count as `len(points)` for validation

Risk:

- multiple source annotation formats must collapse to one wrapper output contract

### URC

Current local truth:

- source labels are `.h5`
- current loader extracts semantic points while avoiding unrelated fields

DM-Count mapping:

- reuse current point extraction logic
- feed only semantic point coordinates into wrapper
- do not mix unrelated `dis`-style fields into counting supervision

Risk:

- upstream `DM-Count` assumes crowd-count datasets; URC requires stronger local semantic filtering than QNRF/NWPU/ShanghaiTech

## Wrapper Design

Recommended wrapper files:

- `external/baselines/dm_count/local_adapters/datasets.py`
- `external/baselines/dm_count/local_adapters/discrete_map.py`

Responsibilities:

- convert current dataset sources into upstream tuple outputs
- generate downsampled discrete maps with ratio `8`
- preserve image names for validation export

## Reuse Plan

Prefer reusing from `pack/`:

- split semantics
- point extraction logic
- image resize and basic transform conventions

Do not reuse directly from `pack/`:

- Gaussian density targets
- attention masks
- `pack.train` batch dictionaries

## Output Contracts To Lock

### Training Wrapper

Must return:

- `image: Tensor[3,H,W]`
- `points: Tensor[N,2]`
- `gt_discrete: Tensor[1,H/8,W/8]`

### Validation Wrapper

Must return:

- `image: Tensor[3,H,W]`
- `count: int or float tensor`
- `name: str`

## Success Criteria

Data adaptation is considered complete when:

- all three datasets can instantiate under a common wrapper API
- at least one dataset completes one train batch and one validation batch through upstream `DM-Count`
- wrapper outputs are shape-compatible with upstream `train_helper.py` and `test.py`
