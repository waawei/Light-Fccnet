# CAN External Adaptation Design

Date: 2026-04-04

## Goal

将 `CAN` 纳入当前横向对比体系，优先采用官方仓库做 `intake / pinning / audit`，在仓库结构可读且风险可控的前提下，再为后续本地适配写出清晰的执行边界。

## Scope

本轮设计只覆盖三件事：

- 确认 `CAN` 的官方或高可信上游仓库
- 完成 vendoring 所需的 pin 元数据和第一轮结构审查
- 明确是否值得继续进入本地适配 plan

本轮不做：

- 不直接把 `CAN` 注册到 `pack.models`
- 不修改 `pack/train.py`
- 不承诺本轮就能跑出正式 `MAE / MSE / MAPE`

## Upstream Strategy

优先使用官方仓库：

- Upstream URL: `https://github.com/weizheliu/Context-Aware-Crowd-Counting`
- 当前可见默认分支：`master`
- 当前可见候选 pin: `d2e4d0425f578e556c1ab6017d326cff20466fad`

本地存放方式与 `DM-Count` 保持一致：

```text
external/baselines/can/
  LOCAL_README.md
  upstream/
```

其中：

- `LOCAL_README.md` 负责记录 URL、branch、commit、scope、非目标和 audit 结论
- `upstream/` 仅保存 pinned working tree，不保留 `.git`

## Recommended Approach

### Option 1: Official Repo First, Then Adapt

先 vendor 官方仓库，记录 pin，做结构和依赖审查。如果结构可读，再单独写本地适配 spec/plan。

优点：

- 最符合论文表述
- 风险暴露更早
- 复用现有 `DM-Count` 流程

缺点：

- 旧环境债可能较重
- 后续适配难度取决于其数据与训练脚本绑定程度

这是推荐路线。

### Option 2: Repo for Reference Only, Reimplement Locally

只把官方仓库作为结构参考，然后在 `pack/` 里写 `CAN-style` 版本。

优点：

- 最终工程最统一

缺点：

- 论文里很难称为 `CAN` 复现
- 容易偏离原方法

不建议作为主路线。

### Option 3: Literature Citation Only

不接代码，只引用论文或现成报告值。

优点：

- 时间成本最低

缺点：

- 横向对比的说服力明显下降

只在上游质量过差时降级采用。

## Audit Questions

在决定是否继续本地适配前，必须回答这几个问题：

1. 训练入口、评估入口、模型定义是否清晰分离。
2. 输入监督是否仍是标准密度图回归，能否映射到 `GWHD / MTC / URC`。
3. 数据集协议是否能用 wrapper 适配，而不是重写核心训练循环。
4. 是否存在过重的历史环境依赖，例如 `Python 2.x` 或极老 `PyTorch`。
5. 模型是否能在不跑完整训练的情况下单独实例化并测复杂度。

## Success Criteria

本轮完成的判定标准：

- `CAN` 官方仓库已 pin 到本地
- `LOCAL_README.md` 已记录完整 pin 元数据
- `STATUS.md` 与 `INTEGRATION_CHECKLIST.md` 已反映 `CAN` 当前状态
- 已形成明确 audit 结论：
  - `continue to local adaptation spec/plan`
  - 或 `downgrade to citation`

## Expected Result Type

若后续结构可顺利适配，则论文中 `CAN` 应先按 `adapted reproduction` 处理，而不是直接宣称 `local reproduction`。
