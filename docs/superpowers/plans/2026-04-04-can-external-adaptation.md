# CAN External Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor and pin the official `CAN` repository, capture a first-pass audit, and decide whether the project should proceed to a local adaptation plan.

**Architecture:** Follow the same external-baseline workflow already used for `DM-Count`: keep upstream code isolated under `external/baselines/can/upstream/`, store all local integration intent in adjacent markdown files, and defer any runtime adaptation until the repository structure and dependency risks are understood.

**Tech Stack:** Git, PowerShell, Markdown, vendored upstream repository layout, existing `external/baselines/` tracking docs.

---

### Task 1: Create CAN Intake Documents

**Files:**
- Create: `docs/superpowers/specs/2026-04-04-can-external-adaptation-design.md`
- Create: `docs/superpowers/plans/2026-04-04-can-external-adaptation.md`
- Create: `external/baselines/can/LOCAL_README.md`

- [ ] **Step 1: Write the design/spec document**

Document the approved strategy:

- official repo first
- vendor and pin before adaptation
- continue only if structure is readable and risks are bounded

- [ ] **Step 2: Write the implementation plan**

Define the intake tasks, audit targets, and stop conditions.

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
Get-ChildItem docs\superpowers\specs\2026-04-04-can-external-adaptation-design.md, docs\superpowers\plans\2026-04-04-can-external-adaptation.md, external\baselines\can\LOCAL_README.md
```

Expected: all three files listed.

- [ ] **Step 5: Commit**

```powershell
git add docs/superpowers/specs/2026-04-04-can-external-adaptation-design.md docs/superpowers/plans/2026-04-04-can-external-adaptation.md external/baselines/can/LOCAL_README.md
git commit -m "docs: add can external adaptation spec and intake notes"
```

### Task 2: Vendor and Pin Official CAN Upstream

**Files:**
- Modify: `external/baselines/can/LOCAL_README.md`
- Create: `external/baselines/can/upstream/`
- Delete: `external/baselines/can/.gitkeep`

- [ ] **Step 1: Resolve upstream identity**

Run:

```powershell
git ls-remote https://github.com/weizheliu/Context-Aware-Crowd-Counting.git HEAD refs/heads/master refs/heads/main
```

Expected: concrete HEAD hash and branch reference.

- [ ] **Step 2: Clone upstream to a temporary directory**

Run:

```powershell
git clone --depth 1 https://github.com/weizheliu/Context-Aware-Crowd-Counting.git <temp-dir>
```

Expected: successful shallow clone.

- [ ] **Step 3: Replace local placeholder with vendored snapshot**

Copy the working tree to `external/baselines/can/upstream/` and ensure `.git` is removed.

- [ ] **Step 4: Record pin metadata**

Update `LOCAL_README.md` with the final branch and commit hash actually vendored.

- [ ] **Step 5: Verify vendor boundary**

Run:

```powershell
Get-ChildItem external\baselines\can\upstream
Test-Path external\baselines\can\upstream\.git
```

Expected:

- upstream files are present
- `.git` path returns `False`

- [ ] **Step 6: Commit**

```powershell
git add external/baselines/can
git commit -m "chore: vendor and pin can upstream baseline"
```

### Task 3: Capture First-Pass Audit and Tracking Status

**Files:**
- Modify: `external/baselines/can/LOCAL_README.md`
- Modify: `external/baselines/STATUS.md`
- Modify: `external/baselines/INTEGRATION_CHECKLIST.md`
- Modify: `docs/superpowers/status/CURRENT_HANDOFF.md`

- [ ] **Step 1: Inspect upstream structure**

Check at minimum:

- `train.py`
- `test.py`
- model definition file
- dataset loading file
- dependency declaration files

- [ ] **Step 2: Record audit conclusions**

Document:

- training entry
- evaluation entry
- expected data format
- known dependency risks
- whether complexity measurement looks feasible
- whether local adaptation planning should continue

- [ ] **Step 3: Update global baseline trackers**

Mark `CAN` as:

- active intake if audit is in progress
- pinned and audited if this task is complete

- [ ] **Step 4: Update handoff**

Switch `CURRENT_HANDOFF.md` so the next session can continue from either:

- `CAN local adaptation spec/plan`
- or downgrade decision if audit fails

- [ ] **Step 5: Run verification**

Run:

```powershell
git diff --cached --stat
Get-Content external\baselines\can\LOCAL_README.md -Encoding utf8
Get-Content external\baselines\STATUS.md -Encoding utf8
Get-Content external\baselines\INTEGRATION_CHECKLIST.md -Encoding utf8
```

Expected: pin metadata and audit notes are consistent across files.

- [ ] **Step 6: Commit**

```powershell
git add external/baselines/can/LOCAL_README.md external/baselines/STATUS.md external/baselines/INTEGRATION_CHECKLIST.md docs/superpowers/status/CURRENT_HANDOFF.md
git commit -m "docs: record can upstream audit and tracking state"
```

### Task 4: Decide Whether to Continue

**Files:**
- Modify: `docs/superpowers/status/CURRENT_HANDOFF.md`
- Optionally Create: `external/baselines/can/LOCAL_ADAPTATION_NOTES.md`
- Optionally Create: `external/baselines/can/DATA_ADAPTATION_PLAN.md`
- Optionally Create: `external/baselines/can/RESULT_EXPORT_PLAN.md`

- [ ] **Step 1: Evaluate audit outcome**

Decision rule:

- continue if repository structure is readable and adaptation risk is bounded
- stop and downgrade if dependencies or code quality make adaptation disproportionate

- [ ] **Step 2: If continue, write local adaptation documents**

Only if Task 3 concludes `continue`, add the same three planning documents used by `DM-Count`.

- [ ] **Step 3: If stop, update trackers to `literature citation`**

Record the downgrade clearly in `STATUS.md`, `INTEGRATION_CHECKLIST.md`, and `CURRENT_HANDOFF.md`.

- [ ] **Step 4: Verify decision is explicit**

Run:

```powershell
Select-String -Path external\baselines\can\LOCAL_README.md, docs\superpowers\status\CURRENT_HANDOFF.md -Pattern "continue|downgrade|citation|adaptation"
```

Expected: the chosen direction is explicitly written down.

- [ ] **Step 5: Commit**

If continuing:

```powershell
git add external/baselines/can docs/superpowers/status/CURRENT_HANDOFF.md external/baselines/STATUS.md external/baselines/INTEGRATION_CHECKLIST.md
git commit -m "docs: define can local adaptation next steps"
```

If stopping:

```powershell
git add external/baselines/can docs/superpowers/status/CURRENT_HANDOFF.md external/baselines/STATUS.md external/baselines/INTEGRATION_CHECKLIST.md
git commit -m "docs: downgrade can to citation after audit"
```
