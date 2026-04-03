# Current Handoff

Date: 2026-04-04

## Current Goal

切换到下一条横向基线：`CAN`。

执行路线已经确定：

- 先找官方或高可信外部仓库
- 先做 intake / pinning / audit
- 再决定本地适配方案

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

## Explicit Non-Goals

当前不要做这些事：

- 不把 `DM-Count` 注册进 `pack.models`
- 不修改 `pack/train.py` 主训练循环去兼容 `DM-Count`
- 不把 `DM-Count` 改写成新的 `DM-Count-style baseline`
- 不把 `DM-Count` 的本地 smoke result 写进论文主表
- 不跳过 `CAN` 的 intake 直接做本地实现

## Next Step

下一目标是 `CAN`，先做：

1. 查找官方或高可信外部仓库
2. 记录 upstream URL / default branch / candidate pin
3. 审查训练入口、评估入口、数据格式和依赖风险
4. 写 `CAN` 的 spec / plan
5. 经确认后再进入 vendor / pinning
