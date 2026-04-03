# DM-Count External Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入官方/高可信 `DM-Count` 仓库，并在不破坏当前 `pack/` 主训练框架的前提下，完成最小本地适配与统一协议验证。

**Architecture:** 保持 `DM-Count` 作为 `external/baselines/dm_count/` 下的独立外部基线，先完成仓库固定、依赖审查、数据适配和结果导出，再考虑更深层统一。当前阶段不把 `DM-Count` 注册进 `pack.models`，而是通过适配层与汇总脚本并入论文对比流程。

**Tech Stack:** Git, Python, PyTorch, YAML, PowerShell, `unittest`, current `pack/` evaluation protocol

---

### Task 1: External Repository Intake

**Files:**
- Modify: `external/baselines/STATUS.md`
- Modify: `external/baselines/INTEGRATION_CHECKLIST.md`
- Create: `external/baselines/dm_count/README.md`

- [ ] **Step 1: Write the failing checklist**

在 `external/baselines/dm_count/README.md` 中列出必须确认但当前未知的事项：

- 官方仓库 URL
- 默认分支或 commit
- 训练入口
- 数据格式
- 验证入口
- 依赖版本

- [ ] **Step 2: Verify checklist is incomplete**

人工确认 `external/baselines/dm_count/` 下尚无这些落地信息。

- [ ] **Step 3: Write minimal implementation**

补齐外部基线接入说明，并在 `STATUS.md` 中新增 `DM-Count` 当前阶段状态。

- [ ] **Step 4: Verify docs are coherent**

人工检查三份文档是否一致描述 `DM-Count` 的外部接入路线。

- [ ] **Step 5: Commit**

```bash
git add external/baselines/STATUS.md external/baselines/INTEGRATION_CHECKLIST.md external/baselines/dm_count/README.md
git commit -m "docs: add dm-count external intake checklist"
```

### Task 2: Clone and Pin Official Repository

**Files:**
- Populate: `external/baselines/dm_count/`
- Modify: `external/baselines/STATUS.md`

- [ ] **Step 1: Write the failing verification**

运行：

```powershell
Test-Path external\baselines\dm_count\.git
```

Expected: `False`

- [ ] **Step 2: Clone repository**

将官方/高可信 `DM-Count` 仓库克隆到：

```powershell
external\baselines\dm_count
```

并记录远程 URL 与当前 commit hash。

- [ ] **Step 3: Verify clone succeeded**

运行：

```powershell
git -C external\baselines\dm_count rev-parse HEAD
```

Expected: 返回固定 commit hash

- [ ] **Step 4: Update status**

在 `STATUS.md` 中记录：

- clone 状态
- pinned commit
- 是否为官方仓库

- [ ] **Step 5: Commit**

```bash
git add external/baselines/STATUS.md external/baselines/dm_count
git commit -m "chore: pin external dm-count baseline repository"
```

### Task 3: Dependency and Entry Audit

**Files:**
- Create: `external/baselines/dm_count/LOCAL_ADAPTATION_NOTES.md`

- [ ] **Step 1: Write the failing checklist**

在 `LOCAL_ADAPTATION_NOTES.md` 中先列空项：

- Python 版本
- PyTorch 版本
- CUDA 需求
- 训练入口文件
- 评估入口文件
- 预期数据目录结构

- [ ] **Step 2: Verify unknowns remain**

人工确认这些信息当前未系统整理。

- [ ] **Step 3: Write minimal implementation**

阅读外部仓库结构并补齐上述字段，给出与当前 `pack/` 仓库的接口映射说明。

- [ ] **Step 4: Verify notes are actionable**

人工检查文档是否足以让后续适配工作在不重新探索仓库的前提下继续。

- [ ] **Step 5: Commit**

```bash
git add external/baselines/dm_count/LOCAL_ADAPTATION_NOTES.md
git commit -m "docs: audit dm-count dependencies and entrypoints"
```

### Task 4: Data Adaptation Design

**Files:**
- Create: `external/baselines/dm_count/DATA_ADAPTATION_PLAN.md`
- Modify: `docs/superpowers/specs/2026-04-03-light-fccnet-horizontal-baseline-reproduction-checklist.md`

- [ ] **Step 1: Write the failing checklist**

列出三个数据集的映射空表：

- `GWHD_2021`
- `MTC`
- `URC`

每个都需要：

- 原始标注形态
- `DM-Count` 期望输入格式
- 需要新增的转换或 wrapper

- [ ] **Step 2: Verify mapping is currently unspecified**

人工确认当前仓库还没有 `DM-Count` 数据映射方案。

- [ ] **Step 3: Write minimal implementation**

补齐数据映射设计，并在横向对比清单中注明 `DM-Count` 当前走“外部仓库 + 本地适配”路线。

- [ ] **Step 4: Verify mapping is consistent**

人工检查数据映射方案与统一协议不冲突。

- [ ] **Step 5: Commit**

```bash
git add external/baselines/dm_count/DATA_ADAPTATION_PLAN.md docs/superpowers/specs/2026-04-03-light-fccnet-horizontal-baseline-reproduction-checklist.md
git commit -m "docs: define dm-count dataset adaptation strategy"
```

### Task 5: Unified Metric and Complexity Hooks

**Files:**
- Create: `external/baselines/dm_count/RESULT_EXPORT_PLAN.md`
- Modify: `external/baselines/README.md`

- [ ] **Step 1: Write the failing checklist**

明确当前 `DM-Count` 缺少的统一协议输出：

- `MAE`
- `MSE`
- `MAPE`
- `Params`
- `Approx FLOPs`

- [ ] **Step 2: Verify hooks are absent**

人工确认当前没有统一导出脚本或复杂度统计方案。

- [ ] **Step 3: Write minimal implementation**

记录结果导出方案和复杂度统计口径，明确哪些将引用现有 `pack/tools/measure_model_complexity.py` 风格，哪些需要外部 wrapper。

- [ ] **Step 4: Verify plan is enough for implementation**

人工确认文档已能指导后续脚本实现。

- [ ] **Step 5: Commit**

```bash
git add external/baselines/dm_count/RESULT_EXPORT_PLAN.md external/baselines/README.md
git commit -m "docs: plan dm-count metric export and complexity hooks"
```

## Verification Commands

在真正进入实现前，应完成下面这些检查：

```powershell
git -C external\baselines\dm_count rev-parse HEAD
Get-Content external\baselines\dm_count\README.md
Get-Content external\baselines\dm_count\LOCAL_ADAPTATION_NOTES.md
Get-Content external\baselines\dm_count\DATA_ADAPTATION_PLAN.md
Get-Content external\baselines\dm_count\RESULT_EXPORT_PLAN.md
```

Expected outcomes:

- `DM-Count` 外部仓库已固定版本
- 依赖、训练入口、数据协议、结果导出方案都有书面说明
- 后续实现工作可以在不重复探索仓库的前提下继续推进
