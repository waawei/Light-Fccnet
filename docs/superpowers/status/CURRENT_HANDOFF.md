# Current Handoff

Date: 2026-04-04

## Current Goal

推进 `SASNet` 到本地适配规划阶段。

当前已经完成：

- 官方仓库 intake
- upstream pinning
- first-pass audit
- local adaptation notes
- data adaptation plan
- result export plan

下一步是：

- 决定是否进入 `SASNet` adapter 实现
- 若进入，实现 dataset / eval / export / complexity wrappers

## Repository Root

`d:\develop\python\SecondChoice`

## Non-Negotiable Context

- 当前主代码目录是 `pack/`
- 不要再新增或恢复旧的 `code/` 路径引用
- Git 远程已可通过 `waawei` 的 SSH 正常推送
- 正式论文性能指标必须来自 `autodl` 完整训练
- 本地结果只用于接口验证、复杂度统计、训练桥接和 smoke test

## Verified Completed State

### Light-FCCNet / CSRNet

以下测试已在本机 `Python 3.11 + torch 2.6.0+cu124` 环境中真实通过：

- `tests.test_light_fccnet`
- `tests.test_csrnet_baseline`
- `tests.test_baseline_complexity`

已测复杂度：

- `Light-FCCNet`: `915,860` params, `2495.49G` approx FLOPs
- `CSRNet`: `16,263,489` params, `1713.44G` approx FLOPs

### DM-Count

`DM-Count` 已完成本地可继续复现的桥接层，当前状态如下：

- upstream URL: `https://github.com/cvlab-stonybrook/DM-Count`
- upstream branch: `master`
- pinned commit: `cc5f2132e0d1328909f31b6d665b8e0b15c30467`
- vendored snapshot path: `external/baselines/dm_count/upstream/`
- upstream `.git` 已移除

本地已补齐：

- intake / pinning 记录
- local adaptation notes
- data adaptation plan
- result export plan
- local dataset adapters
- local eval / export / complexity wrappers
- local run entry
- adapter tests

本地已实测复杂度：

- `DM-Count`: `21,499,457` params, `1706.85G` approx FLOPs

注意：

- `DM-Count` 正式 `MAE / MSE / MAPE` 仍需要上 `autodl` 跑完整训练
- 当前结果类型应标记为 `adapted reproduction`

## Key Files To Read First

新会话继续时，先读这些文件：

1. `docs/superpowers/specs/2026-04-03-light-fccnet-horizontal-baseline-reproduction-checklist.md`
2. `external/baselines/STATUS.md`
3. `external/baselines/INTEGRATION_CHECKLIST.md`
4. `external/baselines/dm_count/LOCAL_README.md`
5. `external/baselines/dm_count/LOCAL_ADAPTATION_NOTES.md`
6. `external/baselines/dm_count/DATA_ADAPTATION_PLAN.md`
7. `external/baselines/dm_count/RESULT_EXPORT_PLAN.md`
8. `external/baselines/can/LOCAL_README.md`
9. `external/baselines/can/LOCAL_ADAPTATION_NOTES.md`
10. `external/baselines/can/DATA_ADAPTATION_PLAN.md`
11. `external/baselines/can/RESULT_EXPORT_PLAN.md`
12. `external/baselines/sasnet/LOCAL_README.md`
13. `external/baselines/sasnet/LOCAL_ADAPTATION_NOTES.md`
14. `external/baselines/sasnet/DATA_ADAPTATION_PLAN.md`
15. `external/baselines/sasnet/RESULT_EXPORT_PLAN.md`

## Explicit Non-Goals

当前不要做这些事：

- 不把 `DM-Count` 注册进 `pack.models`
- 不修改 `pack/train.py` 主训练循环去兼容 `DM-Count`
- 不把 `DM-Count` 改写成新的 `DM-Count-style baseline`
- 不把 `DM-Count` 的本地 smoke result 写进论文主表
- 不跳过 `CAN` 的 intake 直接做本地实现

## CAN Verified Intake State

- upstream URL: `https://github.com/weizheliu/Context-Aware-Crowd-Counting`
- upstream branch: `master`
- pinned commit: `d2e4d0425f578e556c1ab6017d326cff20466fad`
- vendored snapshot path: `external/baselines/can/upstream/`
- upstream `.git` 已移除

第一轮 audit 结论：

- 结构可读，入口清晰，值得继续
- 训练入口：`train.py`
- 测试入口：`test.py`
- 数据集入口：`dataset.py` + `image.py`
- 模型定义：`model.py`
- 主要风险：
  - `Python 2.7`
  - `PyTorch 0.4.1`
  - `xrange`
  - 旧式 `Variable`
  - `/` 整除语义依赖
  - `test.py` 内部硬编码路径

## SASNet Verified Intake State

- upstream URL: `https://github.com/TencentYoutuResearch/CrowdCounting-SASNet`
- upstream branch: `main`
- pinned commit: `3e2b78a6c6ebe761c5be6a9181457daad6df666d`
- vendored snapshot path: `external/baselines/sasnet/upstream/`
- upstream `.git` 已移除

第一轮 audit 结论：

- 结构可读，值得继续
- 主入口：`main.py`
- 模型定义：`model.py`
- 数据准备：`prepare_dataset.py`
- 数据集入口：`datasets/loading_data.py` + `datasets/crowd_dataset.py`
- 主要风险：
  - 官方仓库偏推理而不是完整训练
  - 依赖 `python 3.6.8`、`pytorch >= 1.5.0`
  - 默认目录结构和当前项目协议不一致
  - 需要本地 bridge 才能完成统一训练与结果导出

## Next Step

下一步建议优先顺序：

1. `external/baselines/sasnet/local_adapters/datasets.py`
2. `external/baselines/sasnet/local_adapters/eval.py`
3. `external/baselines/sasnet/local_adapters/export_results.py`
4. `external/baselines/sasnet/local_adapters/measure_complexity.py`
5. 之后再决定是否补 bridge 和 `run_local.py`
