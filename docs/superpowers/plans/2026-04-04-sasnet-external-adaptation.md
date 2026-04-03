# SASNet External Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor and pin the official `SASNet` repository, capture a first-pass audit, and decide whether the project should proceed to a local adaptation plan.

**Architecture:** Follow the same external-baseline workflow already used for `DM-Count` and `CAN`: keep upstream code isolated under `external/baselines/sasnet/upstream/`, store all local integration intent in adjacent markdown files, and defer runtime adaptation until the repository structure and dependency risks are understood.

**Tech Stack:** Git, PowerShell, Markdown, vendored upstream repository layout, existing `external/baselines/` tracking docs.

---

### Task 1: Create SASNet Intake Documents

**Files:**
- Create: `docs/superpowers/specs/2026-04-04-sasnet-external-adaptation-design.md`
- Create: `docs/superpowers/plans/2026-04-04-sasnet-external-adaptation.md`
- Create: `external/baselines/sasnet/LOCAL_README.md`

- [ ] **Step 1: Write the design/spec document**
- [ ] **Step 2: Write the implementation plan**
- [ ] **Step 3: Write `LOCAL_README.md`**

Include:

- upstream URL
- upstream branch
- pinned commit
- pin date
- storage model
- local directory contract
- current scope
- non-goals
- audit targets

- [ ] **Step 4: Verify files exist**

Run:

```powershell
Get-ChildItem docs\superpowers\specs\2026-04-04-sasnet-external-adaptation-design.md, docs\superpowers\plans\2026-04-04-sasnet-external-adaptation.md, external\baselines\sasnet\LOCAL_README.md
```

- [ ] **Step 5: Commit**

```powershell
git add docs/superpowers/specs/2026-04-04-sasnet-external-adaptation-design.md docs/superpowers/plans/2026-04-04-sasnet-external-adaptation.md external/baselines/sasnet/LOCAL_README.md
git commit -m "docs: add sasnet external adaptation spec and intake notes"
```

### Task 2: Vendor and Pin Official SASNet Upstream

**Files:**
- Modify: `external/baselines/sasnet/LOCAL_README.md`
- Create: `external/baselines/sasnet/upstream/`
- Delete: `external/baselines/sasnet/.gitkeep`

- [ ] **Step 1: Resolve upstream identity**

Run:

```powershell
git ls-remote https://github.com/TencentYoutuResearch/CrowdCounting-SASNet.git HEAD refs/heads/main refs/heads/master
```

- [ ] **Step 2: Clone upstream to a temporary directory**
- [ ] **Step 3: Replace local placeholder with vendored snapshot**
- [ ] **Step 4: Record pin metadata**
- [ ] **Step 5: Verify vendor boundary**

Run:

```powershell
Get-ChildItem external\baselines\sasnet\upstream
Test-Path external\baselines\sasnet\upstream\.git
```

- [ ] **Step 6: Commit**

```powershell
git add external/baselines/sasnet
git commit -m "chore: vendor and pin sasnet upstream baseline"
```

### Task 3: Capture First-Pass Audit and Tracking Status

**Files:**
- Modify: `external/baselines/sasnet/LOCAL_README.md`
- Modify: `external/baselines/STATUS.md`
- Modify: `external/baselines/INTEGRATION_CHECKLIST.md`

- [ ] **Step 1: Inspect upstream structure**

Check at minimum:

- `main.py`
- `model.py`
- `prepare_dataset.py`
- `datasets/`
- `requirements.txt`
- `README.md`

- [ ] **Step 2: Record audit conclusions**

Document:

- training or inference entry
- expected data format
- known dependency risks
- whether complexity measurement looks feasible
- whether local adaptation planning should continue

- [ ] **Step 3: Update global baseline trackers**

Mark `SASNet` as:

- active intake if audit is in progress
- pinned and audited if this task is complete

- [ ] **Step 4: Run verification**

Run:

```powershell
git diff --cached --stat
Get-Content external\baselines\sasnet\LOCAL_README.md -Encoding utf8
Get-Content external\baselines\STATUS.md -Encoding utf8
Get-Content external\baselines\INTEGRATION_CHECKLIST.md -Encoding utf8
```

- [ ] **Step 5: Commit**

```powershell
git add external/baselines/sasnet/LOCAL_README.md external/baselines/STATUS.md external/baselines/INTEGRATION_CHECKLIST.md
git commit -m "docs: record sasnet upstream audit and tracking state"
```

### Task 4: Decide Whether to Continue

**Files:**
- Optionally Create: `external/baselines/sasnet/LOCAL_ADAPTATION_NOTES.md`
- Optionally Create: `external/baselines/sasnet/DATA_ADAPTATION_PLAN.md`
- Optionally Create: `external/baselines/sasnet/RESULT_EXPORT_PLAN.md`

- [ ] **Step 1: Evaluate audit outcome**

- [ ] **Step 2: If continue, write local adaptation documents**

Only if Task 3 concludes `continue`, add the same three planning documents used by `DM-Count` and `CAN`.

- [ ] **Step 3: If stop, update trackers to `literature citation`**

- [ ] **Step 4: Verify decision is explicit**

Run:

```powershell
Select-String -Path external\baselines\sasnet\LOCAL_README.md -Pattern "continue|downgrade|citation|adaptation"
```

- [ ] **Step 5: Commit**

If continuing:

```powershell
git add external/baselines/sasnet external/baselines/STATUS.md external/baselines/INTEGRATION_CHECKLIST.md
git commit -m "docs: define sasnet local adaptation next steps"
```

If stopping:

```powershell
git add external/baselines/sasnet external/baselines/STATUS.md external/baselines/INTEGRATION_CHECKLIST.md
git commit -m "docs: downgrade sasnet to citation after audit"
```
