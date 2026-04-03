# SASNet External Adaptation Design

Date: 2026-04-04

## Goal

将 `SASNet` 纳入当前横向对比体系，优先采用官方仓库完成 `intake / pinning / audit`，并在结构可读、迁移风险可控的前提下继续进入本地适配规划。

## Scope

本轮设计只覆盖三件事：

- 确认 `SASNet` 的官方或高可信上游仓库
- 完成 vendoring 所需的 pin 元数据和第一轮结构审查
- 明确是否值得继续进入本地适配 plan

本轮不做：

- 不直接把 `SASNet` 注册到 `pack.models`
- 不修改 `pack/train.py`
- 不承诺本轮就能跑出正式 `MAE / MSE / MAPE`

## Upstream Strategy

优先使用官方仓库：

- Upstream URL: `https://github.com/TencentYoutuResearch/CrowdCounting-SASNet`
- 当前可见默认分支：`main`
- 当前可见候选 pin: `3e2b78a6c6ebe761c5be6a9181457daad6df666d`

本地存放方式与 `DM-Count`、`CAN` 保持一致：

```text
external/baselines/sasnet/
  LOCAL_README.md
  upstream/
```

## Recommended Approach

### Option 1: Official Repo First, Then Adapt

先 vendor 官方仓库，记录 pin，做结构和依赖审查。如果结构可读，再单独写本地适配 spec/plan。

优点：

- 最符合论文表述
- 与 `DM-Count / CAN` 工作流一致
- 风险暴露更早

缺点：

- 数据组织方式和当前项目协议可能存在差异
- 仍然需要处理老版本依赖

这是推荐路线。

### Option 2: Repo for Reference Only, Reimplement Locally

只把官方仓库作为结构参考，然后在 `pack/` 里写 `SASNet-style` 版本。

优点：

- 工程最统一

缺点：

- 论文里很难称为 `SASNet` 复现
- 偏离原方法风险更高

不建议作为主路线。

### Option 3: Literature Citation Only

不接代码，只引用论文或现成结果。

优点：

- 时间成本最低

缺点：

- 横向对比力度明显下降

只在上游质量过差时采用。

## Audit Questions

在决定是否继续本地适配前，必须回答：

1. 训练入口、评估入口、模型定义是否清晰分离。
2. 输入监督是否仍是标准密度图回归，能否映射到 `GWHD / MTC / URC`。
3. 数据准备是否必须依赖离线脚本或特定目录结构。
4. 依赖环境是否仍然在当前机器可控范围内。
5. 模型是否能在不跑完整训练的情况下单独实例化并测复杂度。

## Success Criteria

本轮完成的判定标准：

- `SASNet` 官方仓库已 pin 到本地
- `LOCAL_README.md` 已记录完整 pin 元数据
- `STATUS.md` 与 `INTEGRATION_CHECKLIST.md` 已反映 `SASNet` 当前状态
- 已形成明确 audit 结论：
  - `continue to local adaptation spec/plan`
  - 或 `downgrade to citation`

## Expected Result Type

若后续结构可顺利适配，则论文中 `SASNet` 应先按 `adapted reproduction` 处理，而不是直接宣称 `local reproduction`。
