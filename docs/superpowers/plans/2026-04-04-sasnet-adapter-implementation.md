# SASNet Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal local adapter layer for `SASNet` so the project can build dataset wrappers, compute unified metrics, export result rows, and measure complexity without modifying `pack/train.py` or registering `SASNet` in `pack.models`.

**Architecture:** Keep vendored upstream code isolated under `external/baselines/sasnet/upstream/` and implement project-owned wrappers under `external/baselines/sasnet/local_adapters/`. Mirror the structure already used for `DM-Count` and `CAN` where possible so result export and complexity measurement remain comparable across baselines.

**Tech Stack:** Python 3.11, PyTorch, `pack.data` datasets, `pack.utils.metrics`, `pack.tools.measure_model_complexity`, `unittest`.

---

### Task 1: Add Failing Tests for the SASNet Adapter Surface

**Files:**
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Write the failing tests**

Cover these behaviors:

- `SASNetDatasetAdapter` returns train tuples `(image, density_map)`
- validation adapter returns `(image, count, name)`
- dataset builder supports `gwhd`, `mtc`, `urc`
- metrics helper returns `mae`, `mse`, `mape`
- export helpers write JSON and CSV rows
- complexity wrapper returns positive params and flops without forced pretrained download

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters -v
```

Expected: FAIL with import or missing symbol errors because `external.baselines.sasnet.local_adapters` does not exist yet.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_sasnet_adapters.py
git commit -m "test: add sasnet adapter contract coverage"
```

### Task 2: Implement Density Target Generation and Dataset Adapters

**Files:**
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\__init__.py`
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\density_targets.py`
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\datasets.py`
- Test: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Implement minimal density-target helper**

Add a helper that converts point annotations into `SASNet`-compatible density targets aligned to the upstream `sigma4` convention.

- [ ] **Step 2: Implement `SASNetDatasetAdapter`**

Design contract:

- train split returns `(image, density_map)`
- val/test split returns `(image, count, name)`

- [ ] **Step 3: Implement config-based dataset builder**

Map:

- `gwhd` -> `GWHDDataset`
- `mtc` -> `MTCDataset`
- `urc` -> `URCDataset`

using `pack.data.transforms`.

- [ ] **Step 4: Run the targeted tests**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters.SASNetDatasetAdapterTests tests.test_sasnet_adapters.SASNetDatasetBuilderTests tests.test_sasnet_adapters.SASNetDensityTargetTests -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add external/baselines/sasnet/local_adapters tests/test_sasnet_adapters.py
git commit -m "feat: add sasnet dataset adapters and density targets"
```

### Task 3: Implement Unified Eval and Export Helpers

**Files:**
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\eval.py`
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\export_results.py`
- Test: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Implement `compute_count_metrics`**

Use:

- `pack.utils.metrics.cal_mae`
- `pack.utils.metrics.cal_mse`
- `pack.utils.metrics.cal_mape`

- [ ] **Step 2: Implement `build_result_row`**

Required row fields:

- `method`
- `dataset`
- `mae`
- `mse`
- `mape`
- `params`
- `flops`
- `result_type`
- `checkpoint_path`

- [ ] **Step 3: Implement export helpers**

Support:

- JSON single-row output
- CSV append output

- [ ] **Step 4: Run the targeted tests**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters.SASNetEvalExportTests -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add external/baselines/sasnet/local_adapters tests/test_sasnet_adapters.py
git commit -m "feat: add sasnet metric export helpers"
```

### Task 4: Implement Complexity Wrapper

**Files:**
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\measure_complexity.py`
- Test: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Implement a safe SASNet model builder for complexity**

Instantiate `SASNet` while preventing uncontrolled pretrained-weight download.

- [ ] **Step 2: Implement `measure_sasnet_complexity`**

Use `pack.tools.measure_model_complexity.measure_model` and return:

- `model_name`
- `input_shape`
- `params`
- `flops`

- [ ] **Step 3: Run the targeted tests**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters.SASNetComplexityWrapperTests -v
```

Expected: PASS.

- [ ] **Step 4: Run the full SASNet adapter test file**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add external/baselines/sasnet/local_adapters tests/test_sasnet_adapters.py
git commit -m "feat: add sasnet complexity wrapper"
```

### Task 5: Baseline Verification and Handoff

**Files:**
- Modify: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\docs\superpowers\status\CURRENT_HANDOFF.md`

- [ ] **Step 1: Run regression suite**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters tests.test_dm_count_adapters tests.test_baseline_complexity -v
```

Expected: PASS.

- [ ] **Step 2: Record next-step handoff**

Update `CURRENT_HANDOFF.md` to point to the next `SASNet` slice:

- optional runner bridge
- optional local training smoke test
- formal complexity measurement

- [ ] **Step 3: Commit**

```powershell
git add docs/superpowers/status/CURRENT_HANDOFF.md
git commit -m "docs: update handoff after sasnet adapter implementation"
```
