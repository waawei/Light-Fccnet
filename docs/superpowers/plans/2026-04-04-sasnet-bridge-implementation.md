# SASNet Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `SASNet` local adapter layer with a full local bridge so the project can build dataloaders, train one batch, evaluate counts, and run a local CLI without changing `pack/train.py`.

**Architecture:** Keep vendored upstream code isolated under `external/baselines/sasnet/upstream/` and add bridge code under `external/baselines/sasnet/local_adapters/`. Mirror the `DM-Count` and `CAN` bridge shape, but keep `SASNet`-specific semantics explicit: `block_size` model construction and optional `log_para` scaling for training/evaluation.

**Tech Stack:** Python 3.11, PyTorch, `torch.utils.data.DataLoader`, `yaml`, `argparse`, `unittest`, existing `pack.data` datasets and `SASNet` local adapter helpers.

---

### Task 1: Add Failing Tests for the SASNet Bridge Surface

**Files:**
- Modify: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Write the failing tests**

Cover these bridge behaviors:

- `sasnet_train_collate` stacks train tuples correctly
- `evaluate_sasnet_model` returns unified metrics and sample rows
- `build_sasnet_dataloaders` returns `train` and `val`
- `run_train_batch` returns expected loss terms
- CLI parser accepts `--config`, `--dataset-name`, and `--block-size`

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters.SASNetBridgeTests tests.test_sasnet_adapters.SASNetTrainBridgeTests tests.test_sasnet_adapters.SASNetRunnerCliTests -v
```

Expected: FAIL with import or missing symbol errors because the bridge layer does not exist yet.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_sasnet_adapters.py
git commit -m "test: add sasnet bridge contract coverage"
```

### Task 2: Implement Runner Helpers

**Files:**
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\runner.py`
- Modify: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Implement train collate**

Contract:

- input batch: `[(image, density_map), ...]`
- output: `(images, density_maps)` as stacked tensors

- [ ] **Step 2: Implement evaluation helper**

Contract:

- iterate over `(image, count, name)` validation tuples
- sum predicted density map to count
- if `log_para != 1`, divide predicted count by `log_para`
- emit unified `metrics`, per-sample `rows`, and `summary`

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters.SASNetBridgeTests -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add external/baselines/sasnet/local_adapters/runner.py tests/test_sasnet_adapters.py
git commit -m "feat: add sasnet runner helpers"
```

### Task 3: Implement Train Bridge

**Files:**
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\train_bridge.py`
- Modify: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Implement dataloader builder**

Contract:

- build datasets through `build_sasnet_datasets`
- `train` loader uses `sasnet_train_collate`
- `val` loader keeps batch size `1`
- read `batch_size` and `num_workers` from config training section

- [ ] **Step 2: Implement one-batch train step**

Contract:

- move tensors to device
- multiply targets by `log_para` before loss if `log_para != 1`
- use upstream model output directly as density prediction
- optimize with standard density regression loss
- return `loss`, `density_loss`, `mae`, `mse`

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters.SASNetTrainBridgeTests -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add external/baselines/sasnet/local_adapters/train_bridge.py tests/test_sasnet_adapters.py
git commit -m "feat: add sasnet training bridge"
```

### Task 4: Implement Local CLI Runner

**Files:**
- Create: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\external\baselines\sasnet\local_adapters\run_local.py`
- Modify: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\tests\test_sasnet_adapters.py`

- [ ] **Step 1: Implement CLI parser**

Support:

- `--config`
- `--dataset-name`
- `--device`
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--save-dir`
- `--save-prefix`
- `--eval-only`
- `--checkpoint`
- `--block-size`
- `--log-para`

- [ ] **Step 2: Implement local runner**

Requirements:

- load pack YAML config
- optionally override batch size / num_workers
- instantiate upstream `SASNet(pretrained=False, args=Namespace(block_size=...))`
- run local training loop unless `--eval-only`
- save per-epoch checkpoints
- write JSON and CSV summary rows

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters.SASNetRunnerCliTests -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add external/baselines/sasnet/local_adapters/run_local.py tests/test_sasnet_adapters.py
git commit -m "feat: add sasnet local runner"
```

### Task 5: Regression Verification and Handoff

**Files:**
- Modify: `d:\develop\python\SecondChoice\.worktrees\sasnet-adapters\docs\superpowers\status\CURRENT_HANDOFF.md`

- [ ] **Step 1: Run full relevant regression**

Run:

```powershell
python -m unittest tests.test_sasnet_adapters tests.test_dm_count_adapters tests.test_baseline_complexity -v
```

Expected: PASS.

- [ ] **Step 2: Run local complexity measurement again**

Run:

```powershell
python -c "from external.baselines.sasnet.local_adapters.measure_complexity import measure_sasnet_complexity; print(measure_sasnet_complexity((1,3,1080,1920)))"
```

Expected: returns positive params/flops and `model_name == 'sasnet'`.

- [ ] **Step 3: Update handoff**

Record:

- bridge layer complete
- local CLI available
- next step is a training/eval smoke and then `autodl` execution guidance

- [ ] **Step 4: Commit**

```powershell
git add docs/superpowers/status/CURRENT_HANDOFF.md
git commit -m "docs: update handoff after sasnet bridge implementation"
```
