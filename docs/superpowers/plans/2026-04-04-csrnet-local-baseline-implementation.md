# CSRNet Local Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local `CSRNet` baseline to the current reproduction framework so it can be trained and evaluated under the same dataset splits, metrics, and inference protocol as `Light-FCCNet`.

**Architecture:** Implement `CSRNet` directly inside the current `code/` package rather than depending on an external repository. Reuse the existing dataset adapters, training loop, loss policy, and metric stack so only the model definition, model registry wiring, configs, and a minimal compatibility path need to be added.

**Tech Stack:** Python, PyTorch, YAML configs, `unittest`, current `code.train` pipeline

---

### Task 1: Model Surface and Registry

**Files:**
- Create: `code/models/csrnet.py`
- Modify: `code/models/__init__.py`
- Test: `tests/test_baseline_complexity.py`
- Test: `tests/test_csrnet_baseline.py`

- [ ] **Step 1: Write the failing test**

Add a test that calls `build_model({"model": {"name": "csrnet"}})` and expects a `CSRNet` instance with a `predict_count()` method and a `forward()` contract compatible with the current training loop.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_csrnet_baseline.LightBaselineRegistryTests.test_build_model_supports_csrnet`

Expected: FAIL because `csrnet` is not a supported model name.

- [ ] **Step 3: Write minimal implementation**

Create a minimal `CSRNet` module with:

- a VGG-style front-end,
- dilated back-end layers,
- a `1-channel` density head,
- `forward()` returning `(final_density, attention, raw_density)` to match the current pipeline,
- `attention` as a ones tensor placeholder,
- `predict_count()` summing density.

Update `build_model()` to support `model.name == "csrnet"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_csrnet_baseline.LightBaselineRegistryTests.test_build_model_supports_csrnet`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/models/csrnet.py code/models/__init__.py tests/test_csrnet_baseline.py
git commit -m "feat: add local csrnet baseline registry"
```

### Task 2: Shape and Counting Contract

**Files:**
- Modify: `code/models/csrnet.py`
- Test: `tests/test_csrnet_baseline.py`

- [ ] **Step 1: Write the failing test**

Add tests that:

- run `CSRNet` on a `2 x 3 x 128 x 128` tensor,
- assert returned tensor shapes are `(2, 1, 128, 128)`,
- assert `predict_count()` returns a batch-length vector.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_csrnet_baseline.CSRNetForwardContractTests`

Expected: FAIL on shape or contract mismatch.

- [ ] **Step 3: Write minimal implementation**

Adjust the model so the output is resized to input shape when needed, matching the Light-FCCNet contract.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_csrnet_baseline.CSRNetForwardContractTests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/models/csrnet.py tests/test_csrnet_baseline.py
git commit -m "test: lock csrnet output contract"
```

### Task 3: Config Entry Points

**Files:**
- Create: `code/config/gwhd/config_gwhd_csrnet.yaml`
- Create: `code/config/mtc/config_mtc_csrnet.yaml`
- Create: `code/config/urc/config_urc_csrnet.yaml`
- Test: `tests/test_csrnet_baseline.py`

- [ ] **Step 1: Write the failing test**

Add a config smoke test that loads each CSRNet config and successfully builds a model.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_csrnet_baseline.CSRNetConfigTests`

Expected: FAIL because the config files do not exist.

- [ ] **Step 3: Write minimal implementation**

Create YAML configs mirroring current Light-FCCNet dataset paths and training semantics, but with:

- `model.name: csrnet`
- explicit `input_size`
- loss policy set to the baseline count+density loss unless an ablation requires otherwise.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_csrnet_baseline.CSRNetConfigTests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/config/gwhd/config_gwhd_csrnet.yaml code/config/mtc/config_mtc_csrnet.yaml code/config/urc/config_urc_csrnet.yaml tests/test_csrnet_baseline.py
git commit -m "chore: add csrnet dataset configs"
```

### Task 4: Training Compatibility Smoke Test

**Files:**
- Modify: `tests/test_csrnet_baseline.py`
- Modify: `code/models/csrnet.py` if needed

- [ ] **Step 1: Write the failing test**

Add a lightweight smoke test that imports `code.train`, injects `pack` aliases like the existing tests, and verifies one fake forward-loss pass can execute with `csrnet`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_csrnet_baseline.CSRNetTrainCompatibilityTests`

Expected: FAIL because the training pipeline expects missing fields or shape semantics.

- [ ] **Step 3: Write minimal implementation**

Patch only what is needed so the local baseline fits the current training contract without special-case hacks in `train.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_csrnet_baseline.CSRNetTrainCompatibilityTests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/models/csrnet.py tests/test_csrnet_baseline.py
git commit -m "fix: make csrnet compatible with local training contract"
```

### Task 5: Complexity Measurement Compatibility

**Files:**
- Modify: `code/tools/measure_model_complexity.py`
- Modify: `tests/test_baseline_complexity.py`

- [ ] **Step 1: Write the failing test**

Add a test that measures complexity for `light_fccnet` and `csrnet` from configs and asserts:

- params are positive,
- FLOPs are positive,
- the report includes model name and input shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_baseline_complexity.BaselineComplexityIntegrationTests`

Expected: FAIL because `csrnet` measurement path is not supported yet.

- [ ] **Step 3: Write minimal implementation**

Update the complexity tool so it builds either model through the shared registry and handles the current forward contract.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_baseline_complexity.BaselineComplexityIntegrationTests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/tools/measure_model_complexity.py tests/test_baseline_complexity.py
git commit -m "feat: support csrnet in complexity measurement"
```

### Task 6: Baseline Documentation and Handoff

**Files:**
- Modify: `external/baselines/README.md`
- Modify: `docs/superpowers/specs/2026-04-03-light-fccnet-horizontal-baseline-reproduction-checklist.md`

- [ ] **Step 1: Write the failing test**

No code test. Use a documentation checklist:

- CSRNet status present in baseline registry
- local implementation noted as preferred path
- config names documented
- complexity command documented

- [ ] **Step 2: Verify checklist currently fails**

Open the docs and confirm CSRNet local implementation details are incomplete.

- [ ] **Step 3: Write minimal implementation**

Document:

- why CSRNet is local-first,
- which configs to use,
- the exact complexity measurement command,
- what counts as “reproduced under unified local protocol”.

- [ ] **Step 4: Verify checklist passes**

Manually confirm all four documentation items are present.

- [ ] **Step 5: Commit**

```bash
git add external/baselines/README.md docs/superpowers/specs/2026-04-03-light-fccnet-horizontal-baseline-reproduction-checklist.md
git commit -m "docs: document csrnet local baseline workflow"
```

## Verification Commands

Run the full focused suite after all tasks:

```bash
python -m unittest tests.test_light_fccnet tests.test_baseline_complexity tests.test_csrnet_baseline
python code/tools/measure_model_complexity.py --config code/config/gwhd/config_gwhd_light_full.yaml --input-shape 1 3 1080 1920
python code/tools/measure_model_complexity.py --config code/config/gwhd/config_gwhd_csrnet.yaml --input-shape 1 3 1080 1920
```

Expected outcomes:

- all tests pass,
- both commands print params/FLOPs without crashing,
- no model-specific special-case path is required in the main training loop.
