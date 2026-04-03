# Light-FCCNet Paper Alignment Design

Date: 2026-04-01

## Context

The current `light_fccnet` implementation does not match the paper's ablation semantics.

From the paper pages:

- `P1` is the lightweight feature pyramid aggregation module.
- `P2` is the multi-attention module.
- `P3` is the proposed loss function `L_FCC`.

The current codebase instead uses `use_p1/use_p2/use_p3` as branch-level feature toggles after the backbone. That design makes the experiment names diverge from the paper and weakens the interpretability of ablation results.

This design document defines a paper-aligned refactor so the model structure, training logic, and configuration naming match the thesis figures and ablation table.

## Goals

- Align `P1/P2/P3` with the paper's module definitions.
- Make `baseline`, `+P1`, `+P1+P2`, and `full` correspond to the paper's ablation sequence.
- Separate model structure switches from loss-function switches.
- Keep the code testable by isolating baseline path, pyramid aggregation, attention, and loss policy.

## Non-Goals

- Reproducing exact paper metrics in this phase.
- Reworking dataset preprocessing beyond what is required for paper-aligned training.
- Introducing new modules not described in the paper.
- Preserving the old branch-level meaning of `use_p1/use_p2/use_p3`.

## Paper-Aligned Module Semantics

### P1

`P1` means the lightweight feature pyramid aggregation module.

When enabled:

- the model produces multi-scale features,
- upsamples them to a common spatial resolution,
- and prepares them for downstream fusion.

When disabled:

- the model uses a baseline single-scale path.

### P2

`P2` means the multi-attention module.

It includes:

- spatial attention,
- channel attention,
- and the fusion path that applies those attention operations to pyramid-aligned features.

`P2` is only meaningful when `P1` is enabled. If `P1` is disabled, `P2` must be treated as disabled even if the config sets it to true.

### P3

`P3` means the proposed loss function `L_FCC = (1 - alpha) * (L2 + Lc) + alpha * Ls`.

`P3` is not a structural branch and must not be represented as a feature-path toggle.

## Architecture

### Baseline Path

The baseline path is a real baseline model path, not the full model with zeroed pyramid branches.

Data flow:

`image -> baseline backbone -> baseline projection/fusion -> density head -> density map`

This path provides the reference model used by the paper's ablation study.

### P1-Enhanced Path

When `P1` is enabled, the model replaces the baseline single-scale path with lightweight feature pyramid aggregation.

Data flow:

`image -> backbone -> pyramid aggregation -> aligned multi-scale features -> fusion -> density head`

This path is responsible for the "lightweight feature pyramid aggregation module" improvement claimed by the paper.

### P2-Enhanced Path

When `P2` is enabled on top of `P1`, the aligned pyramid features pass through the multi-attention module before the density head.

Data flow:

`image -> backbone -> pyramid aggregation -> multi-attention -> density head`

The attention module remains a real model component. It is not just logged as an auxiliary map detached from the counting path.

### Output Contract

The model should continue to expose density output suitable for current training and validation utilities. Auxiliary outputs may be kept if they are useful for debugging or supervision, but the main counting output must come from the paper-aligned active path.

## Training and Loss Policy

### Baseline Loss

When `P3` is disabled, training uses a baseline loss policy. The exact baseline loss should be simple and explicit in code. The preferred default is density MSE plus count MSE if that is needed for stable training, but it must be named as a non-paper baseline loss.

### Paper Loss

When `P3` is enabled, training must switch to the paper loss:

- `L2`: pixel-wise density regression loss
- `Lc`: count loss derived from density integration
- `Ls`: structural similarity loss

The training config should treat this as a loss-policy switch rather than a model-branch switch.

### Attention Supervision

The current codebase already produces `attention_mask` in the dataset pipeline. Attention supervision may be retained as an auxiliary engineering aid if needed for training stability, but it must be treated as auxiliary and not redefined as `P3`.

If kept:

- it should be named separately from the paper loss,
- and it should not change the meaning of the paper ablation table.

## Configuration Design

### Canonical Semantics

The canonical paper-aligned controls should be:

- `model.use_p1`
- `model.use_p2`
- `training.use_p3_loss`

Internal behavior:

- `use_p1 = false` means baseline structure.
- `use_p1 = true, use_p2 = false` means baseline plus pyramid aggregation.
- `use_p1 = true, use_p2 = true` means baseline plus pyramid aggregation plus multi-attention.
- `use_p3_loss = true` means train the active structure with `L_FCC`.

### Compatibility Rules

For a transition period, old configs using `model.use_p3` may still be parsed, but they must be normalized internally to `training.use_p3_loss`.

If both are present:

- `training.use_p3_loss` wins.

If `model.use_p2 = true` while `model.use_p1 = false`:

- normalize to `use_p2 = false`,
- and log or document that the setting is invalid under paper semantics.

## Ablation Mapping

The paper-facing experiment mapping should be:

- `baseline`: `P1=off, P2=off, P3=off`
- `baseline_p1`: `P1=on, P2=off, P3=off`
- `baseline_p1_p2`: `P1=on, P2=on, P3=off`
- `full`: `P1=on, P2=on, P3=on`

Configs such as `baseline_p2`, `baseline_p3`, or `baseline_p2_p3` do not correspond to the paper's progressive ablation logic. They may be preserved only as compatibility or exploratory configs and must not be treated as canonical paper experiments.

## Code Structure Plan

The refactor should isolate four responsibilities:

1. Baseline feature path.
2. Pyramid aggregation module.
3. Multi-attention module.
4. Loss policy selection.

Recommended ownership:

- `code/models/light_fccnet.py`
  - orchestrates model assembly and paper-aligned forward path selection
- `code/models/modules/light_pyramid_fusion.py`
  - owns `P1`
- `code/models/modules/light_attention_fusion.py`
  - owns `P2`
- `code/utils/losses.py`
  - owns `L_FCC` and baseline loss policy
- `code/train.py`
  - maps config to loss policy and validates allowed ablation combinations
- `code/config/**`
  - renamed or normalized to paper-aligned meanings

## Testing Strategy

### Structural Tests

Add tests that verify:

- baseline path does not instantiate or execute `P1/P2` behavior,
- `P1` produces multi-scale aligned features,
- `P2` is only active when `P1` is active,
- `full` executes `P1 + P2`.

### Semantics Tests

Add tests that verify:

- `model.use_p3` is no longer treated as a feature branch,
- `training.use_p3_loss` switches the loss policy,
- invalid combinations are normalized or rejected consistently.

### Regression Tests

Add tests that verify:

- canonical config files parse to the intended paper semantics,
- `baseline`, `baseline_p1`, `baseline_p1_p2`, and `full` map to the expected module set,
- training still returns valid metrics and saves checkpoints.

## Migration Strategy

### Canonical Configs

The canonical retained configs should be:

- `*_light_baseline.yaml`
- `*_light_baseline_p1.yaml`
- `*_light_baseline_p1_p2.yaml`
- `*_light_full.yaml`

### Non-Canonical Configs

Configs such as:

- `*_light_baseline_p2.yaml`
- `*_light_baseline_p3.yaml`
- `*_light_baseline_p2_p3.yaml`

should be treated as non-paper exploratory configs. They may remain on disk temporarily, but they should not be used for thesis tables or default experiment scripts.

## Risks

### Risk 1: Metric Shift During Refactor

Changing baseline semantics will alter previous experiment comparability. This is expected and acceptable because previous semantics were not paper-aligned.

### Risk 2: Compatibility Drift

Some helper scripts currently assume `use_p1/use_p2/use_p3` are all model fields. Those scripts must be updated to reflect the new `P3` loss semantics.

### Risk 3: Overfitting the Paper Narrative

The implementation should follow the paper's module boundaries, but not hallucinate unverified details. Where the paper omits exact engineering choices, the code should use explicit defaults and document them as engineering approximations.

## Acceptance Criteria

The refactor is complete when:

- `P1/P2/P3` in code mean the same thing as in the paper.
- `baseline` is a real baseline path, not full-model branch zeroing.
- `P2` cannot exist without `P1`.
- `P3` is implemented as a loss-policy switch.
- canonical configs match the paper ablation ladder.
- tests cover structural semantics and loss semantics.
- result extraction scripts report paper-aligned experiment meanings.

## Next Step

Create an implementation plan for the refactor, then execute it with tests first.
