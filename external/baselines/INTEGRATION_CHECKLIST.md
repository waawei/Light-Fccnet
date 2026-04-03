# Baseline Integration Checklist

This checklist is the execution-facing companion to `external/baselines/STATUS.md`.

## Unified Rules Before Integrating Any Baseline

- [ ] Use the current project dataset splits for `GWHD_2021`, `MTC`, and `URC`
- [ ] Keep metrics fixed to `MAE`, `MSE`, and `MAPE`
- [ ] Keep inference fixed to single-scale evaluation unless the paper comparison explicitly states otherwise
- [ ] Measure complexity with `python code\tools\measure_model_complexity.py --config <config> --input-shape 1 3 1080 1920`
- [ ] Mark each result as one of:
  - local reproduction
  - adapted reproduction
  - literature citation

## Reference PDFs

All paper PDFs currently available locally:

- `D:\develop\python\SecondChoice\ExperimentReference\CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting.pdf`
- `D:\develop\python\SecondChoice\ExperimentReference\CSRNet.pdf`
- `D:\develop\python\SecondChoice\ExperimentReference\Liu_Context-Aware_Crowd_Counting_CVPR_2019_paper.pdf`
- `D:\develop\python\SecondChoice\ExperimentReference\NeurIPS-2020-distribution-matching-for-crowd-counting-Paper.pdf`
- `D:\develop\python\SecondChoice\ExperimentReference\Scale Selection for Crowd Counting.pdf`

## Method-by-Method Checklist

### CSRNet

- [ ] Read `ExperimentReference\CSRNet.pdf`
- [ ] Implement locally in `code/models/csrnet.py`
- [ ] Add configs for `GWHD`, `MTC`, `URC`
- [ ] Verify training loop compatibility
- [ ] Measure local Params/FLOPs
- [ ] Mark final result as `local reproduction`

### DM-Count

- [ ] Read `ExperimentReference\NeurIPS-2020-distribution-matching-for-crowd-counting-Paper.pdf`
- [ ] Search for official repository and record commit/URL
- [ ] Clone into `external/baselines/dm_count/` if repository quality is acceptable
- [ ] Adapt dataset loading to current project splits
- [ ] Export metrics under local protocol
- [ ] Measure local Params/FLOPs if the model can be instantiated cleanly
- [ ] Mark final result as `adapted reproduction` unless reproduced with zero structural changes

### CAN

- [ ] Read `ExperimentReference\Liu_Context-Aware_Crowd_Counting_CVPR_2019_paper.pdf`
- [ ] Search for official or trusted implementation
- [ ] Decide between external clone and local structural implementation
- [ ] Align output contract with current counting pipeline
- [ ] Measure local Params/FLOPs
- [ ] Mark final result as `adapted reproduction`

### SASNet

- [ ] Read `ExperimentReference\Scale Selection for Crowd Counting.pdf`
- [ ] Confirm official repository availability
- [ ] Delay integration until `CSRNet`, `DM-Count`, and `CAN` are stable
- [ ] Adapt to local protocol only after primary baselines are complete
- [ ] Mark final result as `adapted reproduction` or `literature citation`

### CMTL

- [ ] Read `ExperimentReference\CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting.pdf`
- [ ] Confirm whether official code is still obtainable and runnable
- [ ] If repository quality is poor, stop and downgrade to `literature citation`
- [ ] Only integrate if the time cost stays bounded

### M-SegNet

- [ ] Do not start repository integration under the mainline counting protocol
- [ ] Keep as `literature citation` or appendix-only discussion unless the paper text is rewritten to justify a segmentation-style adapted baseline

## Immediate Work Order

1. `CSRNet`
2. `DM-Count`
3. `CAN`
4. `SASNet`
5. `CMTL`
6. `M-SegNet` (citation only by default)
