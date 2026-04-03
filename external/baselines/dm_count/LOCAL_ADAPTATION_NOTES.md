# DM-Count Local Adaptation Notes

Date: 2026-04-04

## Scope

本文件记录 `DM-Count` 官方仓库与当前 `SecondChoice` 仓库之间的真实接缝，用于指导后续最小适配实现。

## Upstream Runtime Facts

- Training entry: `external/baselines/dm_count/upstream/train.py`
- Trainer implementation: `external/baselines/dm_count/upstream/train_helper.py`
- Evaluation entry: `external/baselines/dm_count/upstream/test.py`
- Dataset module: `external/baselines/dm_count/upstream/datasets/crowd.py`
- Model entry: `external/baselines/dm_count/upstream/models.py`
- Requirements file: `external/baselines/dm_count/upstream/requirements.txt`

## Upstream CLI Contract

### Train

```text
python train.py --dataset <qnrf|nwpu|sha|shb> --data-dir <dataset_root> --device <gpu_id>
```

Important defaults:

- optimizer: `Adam`
- lr: `1e-5`
- weight decay: `1e-4`
- max epoch: `1000`
- validation starts at epoch `50`
- validation frequency defaults to every `5` epochs

Dataset-specific crop overrides:

- `qnrf`: `512`
- `nwpu`: `384`
- `sha`: `256`
- `shb`: `512`

### Validation / Test

```text
python test.py --model-path <ckpt> --data-path <dataset_root> --dataset <qnrf|nwpu|sha|shb>
```

Reported metrics:

- `mae`
- `mse`

Not reported upstream:

- `MAPE`

## Upstream Data Contract

### Training Batch

`train_helper.py` expects:

- `image`
- `points`
- `gt_discrete`

More precisely:

- `image`: stacked tensor `[B, 3, H, W]`
- `points`: variable-length list of point tensors
- `gt_discrete`: stacked downsampled discrete maps `[B, 1, H/8, W/8]`

### Validation Batch

`train_helper.py` and `test.py` expect:

- `image`
- `count`
- `name`

This means the upstream validation path does not require density maps, only image-level count labels.

## Upstream Annotation Assumptions

Implemented dataset classes:

- `Crowd_qnrf`
- `Crowd_nwpu`
- `Crowd_sh`

Annotation formats:

- `.jpg + .npy` point arrays for QNRF and NWPU
- `.mat` for ShanghaiTech

Core semantic assumption:

- point locations are represented in image pixel coordinates
- train-time supervision uses a discrete point map downsampled by `8`
- count is recovered from point count or density sum

## Local Repository Contract

Current `pack/` datasets produce dictionaries with:

- `image`
- `density`
- `attention_mask`
- `count`
- `points`
- `image_name`

This differs from upstream `DM-Count`, which expects tuple-style batches and downsampled discrete maps for training.

## Required Adaptation Bridge

The adaptation layer needs to do three things:

1. Reuse current dataset splits and semantic point labels from `pack/`
2. Convert point supervision to upstream-compatible tuple batches
3. Generate `gt_discrete` as a downsampled discrete map, not a Gaussian density map

## Compatibility Constraints

- First implementation should avoid modifying upstream loss logic
- First implementation should avoid modifying upstream model definition
- First implementation may patch CLI defaults, dataset names, and path resolution
- Any wrapper should clearly distinguish `local reproduction` from `adapted reproduction`

## Immediate Conclusion

The safest next implementation slice is:

- add a local dataset wrapper around current `GWHD / MTC / URC` point labels
- keep upstream trainer and loss computation intact
- add a local evaluation/export layer for `MAPE` and paper-table fields
