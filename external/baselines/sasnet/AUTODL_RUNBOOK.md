# SASNet AutoDL Runbook

Date: 2026-04-04

## Purpose

This runbook records the recommended `autodl` commands for running the local `SASNet` bridge on `GWHD`, `MTC`, and `URC`.

## Preconditions

- Run from repository root:
  - `d:\develop\python\SecondChoice` locally
  - `/root/autodl-tmp/SecondChoice` or equivalent on `autodl`
- Use the vendored/local bridge entry:
  - `external/baselines/sasnet/local_adapters/run_local.py`
- Formal paper-facing `MAE / MSE / MAPE` must come from these full runs, not local smoke tests

## Notes

- `SASNet` bridge currently defaults to `--log-para 1000`, following the official upstream inference script
- `--block-size 32` matches the current upstream default
- Use `--device cuda` on `autodl`
- Save outputs into a dataset-specific directory so checkpoints and summaries do not overwrite each other

## GWHD

```bash
python external/baselines/sasnet/local_adapters/run_local.py \
  --config pack/config/gwhd/config_gwhd_light_full.yaml \
  --dataset-name gwhd \
  --device cuda \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 4 \
  --block-size 32 \
  --log-para 1000 \
  --save-dir external/baselines/sasnet/runs/gwhd \
  --save-prefix sasnet_gwhd
```

## MTC

```bash
python external/baselines/sasnet/local_adapters/run_local.py \
  --config pack/config/mtc/config_mtc_light_full.yaml \
  --dataset-name mtc \
  --device cuda \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 4 \
  --block-size 32 \
  --log-para 1000 \
  --save-dir external/baselines/sasnet/runs/mtc \
  --save-prefix sasnet_mtc
```

## URC

```bash
python external/baselines/sasnet/local_adapters/run_local.py \
  --config pack/config/urc/config_urc_light_full.yaml \
  --dataset-name urc \
  --device cuda \
  --epochs 100 \
  --batch-size 6 \
  --num-workers 4 \
  --block-size 32 \
  --log-para 1000 \
  --save-dir external/baselines/sasnet/runs/urc \
  --save-prefix sasnet_urc
```

## Eval-Only Pattern

If a checkpoint already exists and only evaluation is needed:

```bash
python external/baselines/sasnet/local_adapters/run_local.py \
  --config <pack-config> \
  --dataset-name <gwhd|mtc|urc> \
  --device cuda \
  --eval-only \
  --checkpoint <checkpoint-path> \
  --block-size 32 \
  --log-para 1000 \
  --save-dir <output-dir> \
  --save-prefix <run-name>
```

## Expected Outputs

Each run writes:

- epoch checkpoints:
  - `external/baselines/sasnet/runs/<dataset>/<prefix>_<dataset>_epoch*.pth`
- JSON summary:
  - `external/baselines/sasnet/runs/<dataset>/<prefix>_<dataset>_summary.json`
- CSV summary:
  - `external/baselines/sasnet/runs/<dataset>/<prefix>_<dataset>_summary.csv`

## Local Limitation

On the current Windows workstation, the pack configs point to `/root/autodl-tmp/...` dataset mounts, so a true local data-backed smoke run is not possible without rewriting config paths. CLI startup has been verified locally with:

```powershell
python -m external.baselines.sasnet.local_adapters.run_local --help
```
