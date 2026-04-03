# Light-FCCNet Paper Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `light_fccnet` so `P1/P2/P3` match the paper's module semantics, canonical ablation configs match the thesis ladder, and training scripts report paper-aligned experiment meanings.

**Architecture:** Split the current implementation into a real baseline path, a paper-aligned `P1` pyramid aggregation path, a paper-aligned `P2` multi-attention enhancement, and a `P3` loss-policy switch in training. Preserve compatibility where feasible, but make canonical configs and result scripts reflect the paper's ablation semantics rather than the old branch-zeroing semantics.

**Tech Stack:** Python 3.11, PyTorch, unittest, YAML configs

---

## File Map

**Primary files to modify**

- `d:\develop\python\SecondChoice\code\models\light_fccnet.py`
  - Rebuild top-level model assembly around baseline path, `P1`, and `P2`.
- `d:\develop\python\SecondChoice\code\models\__init__.py`
  - Normalize config semantics and map legacy `use_p3` into training loss policy.
- `d:\develop\python\SecondChoice\code\models\modules\light_pyramid_fusion.py`
  - Own the `P1` path and expose a clear interface for aligned multi-scale features.
- `d:\develop\python\SecondChoice\code\models\modules\light_attention_fusion.py`
  - Own the `P2` path and allow bypass when attention is disabled.
- `d:\develop\python\SecondChoice\code\utils\losses.py`
  - Separate paper loss from non-paper baseline loss policy.
- `d:\develop\python\SecondChoice\code\train.py`
  - Switch loss policy based on `training.use_p3_loss` and validate `P1/P2/P3` combinations.
- `d:\develop\python\SecondChoice\code\tools\extract_best_results.py`
  - Report paper-aligned experiment meaning in exported summaries.
- `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`
  - Add semantic and structural regression coverage.

**Config files to modify**

- Canonical:
  - `d:\develop\python\SecondChoice\code\config\gwhd\config_gwhd_light_baseline.yaml`
  - `d:\develop\python\SecondChoice\code\config\gwhd\config_gwhd_light_baseline_p1.yaml`
  - `d:\develop\python\SecondChoice\code\config\gwhd\config_gwhd_light_baseline_p1_p2.yaml`
  - `d:\develop\python\SecondChoice\code\config\gwhd\config_gwhd_light_full.yaml`
  - `d:\develop\python\SecondChoice\code\config\mtc\config_mtc_light_baseline.yaml`
  - `d:\develop\python\SecondChoice\code\config\mtc\config_mtc_light_baseline_p1.yaml`
  - `d:\develop\python\SecondChoice\code\config\mtc\config_mtc_light_baseline_p1_p2.yaml`
  - `d:\develop\python\SecondChoice\code\config\mtc\config_mtc_light_full.yaml`
  - `d:\develop\python\SecondChoice\code\config\urc\config_urc_light_baseline.yaml`
  - `d:\develop\python\SecondChoice\code\config\urc\config_urc_light_baseline_p1.yaml`
  - `d:\develop\python\SecondChoice\code\config\urc\config_urc_light_baseline_p1_p2.yaml`
  - `d:\develop\python\SecondChoice\code\config\urc\config_urc_light_full.yaml`

**Optional compatibility files to touch**

- Non-canonical configs that currently encode impossible paper states:
  - `*_light_baseline_p2.yaml`
  - `*_light_baseline_p3.yaml`
  - `*_light_baseline_p2_p3.yaml`
  - `*_light_baseline_p1_p3.yaml`
  - `*_light_baseline_p1_p2_p3` is represented by `full`

---

### Task 1: Lock Paper Semantics in Tests

**Files:**
- Modify: `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`
- Test: `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`

- [ ] **Step 1: Write the failing tests for paper semantics**

```python
def test_build_model_treats_p2_without_p1_as_disabled(self):
    cfg = {
        "model": {"name": "light_fccnet", "use_p1": False, "use_p2": True},
        "training": {"use_p3_loss": False},
    }
    model = build_model(cfg)
    self.assertFalse(model.use_p1)
    self.assertFalse(model.use_p2)


def test_build_criterion_uses_paper_loss_only_when_training_use_p3_loss_is_true(self):
    train_module = importlib.import_module("code.train")
    paper = train_module.build_criterion({"training": {"loss_type": "light_fcc", "use_p3_loss": True}})
    baseline = train_module.build_criterion({"training": {"loss_type": "light_fcc", "use_p3_loss": False}})
    self.assertNotEqual(type(paper).__name__, type(baseline).__name__)


def test_light_fccnet_baseline_path_does_not_use_pyramid_outputs(self):
    cfg = {"model": {"name": "light_fccnet", "use_p1": False, "use_p2": False}}
    model = build_model(cfg)
    self.assertFalse(model.use_p1)
    self.assertFalse(model.use_p2)
```

- [ ] **Step 2: Run the new paper-semantics tests and verify they fail**

Run:

```powershell
python -m unittest `
  tests.test_light_fccnet.LightFCCNetBuildTests.test_build_model_treats_p2_without_p1_as_disabled `
  tests.test_light_fccnet.LightFCCLossTests.test_build_criterion_uses_paper_loss_only_when_training_use_p3_loss_is_true `
  -v
```

Expected: failures showing current code still treats `P2` as independent and `P3` as a model-branch concept.

- [ ] **Step 3: Add one failing config regression test for canonical experiment naming**

```python
def test_canonical_configs_follow_paper_ablation_ladder(self):
    canonical = {
        "config_gwhd_light_baseline.yaml": (False, False, False),
        "config_gwhd_light_baseline_p1.yaml": (True, False, False),
        "config_gwhd_light_baseline_p1_p2.yaml": (True, True, False),
        "config_gwhd_light_full.yaml": (True, True, True),
    }
```

- [ ] **Step 4: Run that config test and verify it fails**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightResultToolTests.test_canonical_configs_follow_paper_ablation_ladder -v
```

Expected: failure because current configs and parser still use old semantics.

- [ ] **Step 5: Commit**

If the workspace is later moved into a git repo, commit with:

```bash
git add tests/test_light_fccnet.py
git commit -m "test: lock light-fccnet paper semantics"
```

### Task 2: Refactor Model Assembly into Baseline, P1, and P2

**Files:**
- Modify: `d:\develop\python\SecondChoice\code\models\light_fccnet.py`
- Modify: `d:\develop\python\SecondChoice\code\models\modules\light_pyramid_fusion.py`
- Modify: `d:\develop\python\SecondChoice\code\models\modules\light_attention_fusion.py`
- Modify: `d:\develop\python\SecondChoice\code\models\__init__.py`
- Test: `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`

- [ ] **Step 1: Write one more failing structural test for the baseline path**

```python
def test_light_fccnet_baseline_forward_uses_single_scale_path(self):
    cfg = {"model": {"name": "light_fccnet", "use_p1": False, "use_p2": False}}
    model = build_model(cfg)
    x = torch.randn(2, 3, 64, 64)
    final_density, _, _ = model(x)
    self.assertEqual(tuple(final_density.shape), (2, 1, 64, 64))
```

- [ ] **Step 2: Run the structural baseline test and verify it fails for the right reason**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightFCCNetBuildTests.test_light_fccnet_baseline_forward_uses_single_scale_path -v
```

Expected: failure because the current model still routes through pyramid-zeroing logic instead of a distinct baseline path.

- [ ] **Step 3: Implement the minimal model refactor**

Required code shape:

```python
class LightFCCNet(nn.Module):
    def __init__(..., use_p1=True, use_p2=True):
        self.use_p1 = bool(use_p1)
        self.use_p2 = bool(use_p2) and self.use_p1
        self.backbone = ...
        self.pyramid = LightPyramidFeatureAggregation(...) if self.use_p1 else None
        self.attention_fusion = LightMultiAttentionFusion(...) if self.use_p2 else None
        self.baseline_proj = nn.Sequential(...)

    def forward(self, x):
        base_feat = self.backbone(...)
        if not self.use_p1:
            fused = self.baseline_proj(base_feat)
        else:
            pyramid_feats = self.pyramid(base_feat)
            fused = pyramid_feats if not self.use_p2 else self.attention_fusion(pyramid_feats)[0]
        density = self.density_head(fused)
        return density, attention_map, raw_density
```

- [ ] **Step 4: Normalize config parsing in `code/models/__init__.py`**

Required behavior:

```python
use_p1 = bool(model_cfg.get("use_p1", False))
use_p2 = bool(model_cfg.get("use_p2", False)) and use_p1
legacy_use_p3 = model_cfg.get("use_p3")
if "training" in config and "use_p3_loss" not in config["training"] and legacy_use_p3 is not None:
    config["training"]["use_p3_loss"] = bool(legacy_use_p3)
```

- [ ] **Step 5: Run the targeted build/model tests**

Run:

```powershell
python -m unittest `
  tests.test_light_fccnet.LightFCCNetBuildTests `
  tests.test_light_fccnet.LightBackboneTests `
  -v
```

Expected: all structural tests pass with `P2` disabled automatically when `P1` is off.

- [ ] **Step 6: Commit**

```bash
git add code/models/light_fccnet.py code/models/__init__.py code/models/modules/light_pyramid_fusion.py code/models/modules/light_attention_fusion.py tests/test_light_fccnet.py
git commit -m "refactor: align light-fccnet P1 and P2 with paper semantics"
```

### Task 3: Split Paper Loss from Baseline Loss Policy

**Files:**
- Modify: `d:\develop\python\SecondChoice\code\utils\losses.py`
- Modify: `d:\develop\python\SecondChoice\code\train.py`
- Test: `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`

- [ ] **Step 1: Write failing tests for baseline loss vs paper loss**

```python
def test_build_criterion_returns_baseline_loss_when_use_p3_loss_is_false(self):
    train_module = importlib.import_module("code.train")
    criterion = train_module.build_criterion({"training": {"use_p3_loss": False}})
    self.assertEqual(type(criterion).__name__, "BaselineCountingLoss")


def test_build_criterion_returns_light_fcc_loss_when_use_p3_loss_is_true(self):
    train_module = importlib.import_module("code.train")
    criterion = train_module.build_criterion({"training": {"use_p3_loss": True}})
    self.assertEqual(type(criterion).__name__, "LightFCCLoss")
```

- [ ] **Step 2: Run the criterion tests and verify they fail**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightFCCLossTests -v
```

Expected: failure because `build_criterion` currently always returns `LightFCCLoss`.

- [ ] **Step 3: Implement a minimal non-paper baseline loss**

Required code shape:

```python
class BaselineCountingLoss(nn.Module):
    def forward(self, pred_density, gt_density, gt_count=None):
        l2 = self.mse(pred_density, gt_density)
        count = ...
        total = l2 + count
        return total, {"loss": total, "l2": l2, "count": count}
```

- [ ] **Step 4: Switch `build_criterion()` to use `training.use_p3_loss`**

Required behavior:

```python
use_p3_loss = bool(train_cfg.get("use_p3_loss", False))
if use_p3_loss:
    return LightFCCLoss(...)
return BaselineCountingLoss(...)
```

- [ ] **Step 5: Keep auxiliary attention supervision separate from `P3`**

Required naming:

```python
criterion.attention_loss_weight = float(train_cfg.get("attention_loss_weight", 0.0))
```

Do not let this alter the experiment meaning of `P3`.

- [ ] **Step 6: Run loss and train smoke tests**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightFCCLossTests -v
```

Expected: all loss-related tests pass and the train smoke test still completes.

- [ ] **Step 7: Commit**

```bash
git add code/utils/losses.py code/train.py tests/test_light_fccnet.py
git commit -m "refactor: separate baseline loss from paper P3 loss"
```

### Task 4: Migrate Canonical Configs to Paper Semantics

**Files:**
- Modify: canonical config YAMLs under `d:\develop\python\SecondChoice\code\config\gwhd`
- Modify: canonical config YAMLs under `d:\develop\python\SecondChoice\code\config\mtc`
- Modify: canonical config YAMLs under `d:\develop\python\SecondChoice\code\config\urc`
- Test: `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`

- [ ] **Step 1: Write a failing config test for all canonical files**

```python
def test_canonical_training_configs_parse_to_paper_semantics(self):
    expected = {
        "light_baseline": {"use_p1": False, "use_p2": False, "use_p3_loss": False},
        "light_baseline_p1": {"use_p1": True, "use_p2": False, "use_p3_loss": False},
        "light_baseline_p1_p2": {"use_p1": True, "use_p2": True, "use_p3_loss": False},
        "light_full": {"use_p1": True, "use_p2": True, "use_p3_loss": True},
    }
```

- [ ] **Step 2: Run the config regression test and verify it fails**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightResultToolTests.test_canonical_training_configs_parse_to_paper_semantics -v
```

Expected: failure because canonical configs still carry old `use_p3` model semantics.

- [ ] **Step 3: Update canonical YAMLs**

Required pattern:

```yaml
model:
  use_p1: true
  use_p2: true

training:
  use_p3_loss: true
  loss_type: light_fcc
```

For baseline:

```yaml
model:
  use_p1: false
  use_p2: false

training:
  use_p3_loss: false
```

- [ ] **Step 4: Mark or isolate non-canonical configs**

Minimal acceptable approach:

- leave them parseable,
- ensure scripts do not treat them as canonical paper ablations,
- and document them as exploratory compatibility configs.

- [ ] **Step 5: Run all config-related tests**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightResultToolTests tests.test_light_fccnet.LightTrainingEntrypointTests -v
```

Expected: config parsing and default entrypoint tests pass.

- [ ] **Step 6: Commit**

```bash
git add code/config tests/test_light_fccnet.py
git commit -m "chore: migrate canonical light-fccnet configs to paper semantics"
```

### Task 5: Update Result Extraction and Reporting Semantics

**Files:**
- Modify: `d:\develop\python\SecondChoice\code\tools\extract_best_results.py`
- Modify: `d:\develop\python\SecondChoice\code\tools\extract_light_fccnet_results.py`
- Test: `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`

- [ ] **Step 1: Write a failing result-tool test**

```python
def test_extract_best_results_reads_use_p3_loss_from_training_semantics(self):
    payload = {
        "config": {
            "model": {"name": "light_fccnet", "use_p1": True, "use_p2": True},
            "training": {"use_p3_loss": True},
        }
    }
```

- [ ] **Step 2: Run the result-tool test and verify it fails**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightResultToolTests.test_extract_best_results_reads_use_p3_loss_from_training_semantics -v
```

Expected: failure because extractor currently reads `use_p3` from model semantics.

- [ ] **Step 3: Implement minimal extractor normalization**

Required behavior:

```python
use_p1 = bool(model_cfg.get("use_p1", False))
use_p2 = bool(model_cfg.get("use_p2", False)) and use_p1
use_p3 = bool(training_cfg.get("use_p3_loss", model_cfg.get("use_p3", False)))
```

- [ ] **Step 4: Preserve legacy checkpoints**

If an older checkpoint only has `model.use_p3`, keep reading it as fallback for backward compatibility.

- [ ] **Step 5: Run result-tool tests**

Run:

```powershell
python -m unittest tests.test_light_fccnet.LightResultToolTests -v
```

Expected: result export reflects paper-aligned semantics.

- [ ] **Step 6: Commit**

```bash
git add code/tools/extract_best_results.py code/tools/extract_light_fccnet_results.py tests/test_light_fccnet.py
git commit -m "fix: report light-fccnet ablations using paper semantics"
```

### Task 6: Full Verification Pass

**Files:**
- Modify: any touched files above if verification reveals issues
- Test: `d:\develop\python\SecondChoice\tests\test_light_fccnet.py`

- [ ] **Step 1: Run the full unittest suite**

Run:

```powershell
python -m unittest tests.test_light_fccnet -v
```

Expected: all tests pass.

- [ ] **Step 2: Run a small dynamic sanity check for paper-aligned gradients**

Run:

```powershell
@'
import torch
from code.models import build_model
from code.train import build_criterion

cfg = {
    "model": {"name": "light_fccnet", "use_p1": True, "use_p2": True},
    "training": {"use_p3_loss": True, "loss_type": "light_fcc"},
}
model = build_model(cfg)
criterion = build_criterion(cfg)
x = torch.randn(2, 3, 64, 64)
y = torch.rand(2, 1, 64, 64)
c = torch.rand(2)
pred, att, raw = model(x)
loss, _ = criterion(pred, y, gt_count=c)
loss.backward()
for name, p in model.named_parameters():
    if name in ("attention_fusion.fuse.0.weight", "density_head.conv1.weight"):
        print(name, float(p.grad.abs().mean().item()))
'@ | python -
```

Expected: non-zero gradients for active paper modules.

- [ ] **Step 3: Verify canonical config ladder manually**

Check:

- baseline = `P1 off, P2 off, P3 off`
- baseline_p1 = `P1 on, P2 off, P3 off`
- baseline_p1_p2 = `P1 on, P2 on, P3 off`
- full = `P1 on, P2 on, P3 on`

- [ ] **Step 4: Update plan/spec cross-reference if needed**

If implementation diverges from the design document for a valid reason, update:

- `d:\develop\python\SecondChoice\docs\superpowers\specs\2026-04-01-light-fccnet-paper-alignment-design.md`

- [ ] **Step 5: Final commit**

```bash
git add code tests docs
git commit -m "refactor: align light-fccnet implementation with paper ablations"
```

## Review Note

The writing-plans skill requests a plan-reviewer loop. In this session, the conversation constraints do not allow spawning a review subagent without explicit user delegation. Use human review for this plan document before execution, then choose an execution mode.
