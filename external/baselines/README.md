# External Baselines

This directory is the landing area for third-party baseline repositories and baseline-specific notes used in the Light-FCCNet horizontal comparison study.

## Scope

Only keep items here that are needed for one of these purposes:

- clone and pin an official baseline repository,
- record adaptation notes,
- track reproducibility status,
- store small non-generated metadata files that help integrate a baseline into the current project.

Do not copy large datasets, checkpoints, or generated experiment outputs into this directory.

## Directory Layout

```text
external/
  baselines/
    README.md
    STATUS.md
    cmtl/
    csrnet/
    can/
    dm_count/
    sasnet/
    m_segnet/
```

## Baseline Policy

- `csrnet`: prefer a local implementation inside `code/models/` instead of a hard dependency on an external repository.
- `dm_count`: preferred external clone candidate.
- `can`: external clone if the official or trusted implementation is stable; otherwise fall back to a local structural reimplementation.
- `sasnet`: clone only after the main baselines are stable.
- `cmtl`: investigate repository quality before investing in adaptation.
- `m_segnet`: do not treat as a mainline local reproduction target under the unified counting protocol.

## Immediate Next Commands

Create this structure first:

```powershell
New-Item -ItemType Directory -Force external\baselines\cmtl
New-Item -ItemType Directory -Force external\baselines\csrnet
New-Item -ItemType Directory -Force external\baselines\can
New-Item -ItemType Directory -Force external\baselines\dm_count
New-Item -ItemType Directory -Force external\baselines\sasnet
New-Item -ItemType Directory -Force external\baselines\m_segnet
```

## Measurement Protocol

All baselines that enter the main complexity table should be measured under the same local protocol:

- input tensor shape: `1 x 3 x 1080 x 1920`
- eval mode
- single-scale inference
- report both `Params` and `FLOPs`

Use:

```powershell
python code\tools\measure_model_complexity.py --config <config_path> --input-shape 1 3 1080 1920
```

## CSRNet Local Path

`CSRNet` is now wired as a local baseline implementation through the shared registry instead of an external repository dependency.

Available unified-protocol configs:

- `code/config/gwhd/config_gwhd_csrnet.yaml`
- `code/config/mtc/config_mtc_csrnet.yaml`
- `code/config/urc/config_urc_csrnet.yaml`

Recommended complexity command:

```powershell
python code\tools\measure_model_complexity.py --config code\config\gwhd\config_gwhd_csrnet.yaml --input-shape 1 3 1080 1920
```

This path counts as `reproduced under unified local protocol` because it reuses the current project dataset adapters, metrics, and training loop contract.
