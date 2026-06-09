# AGENTS.md — EFAST

## Project Overview

EFAST fuses Sentinel-2 (high-resolution, infrequent) and Sentinel-3 (low-resolution, daily) satellite imagery to produce cloud-free, high-resolution time-series images. The same approach works for Landsat/MODIS pairs. The core algorithm is described in [Senty et al. (2024)](https://doi.org/10.3390/rs16111833).

## Repository Layout

```
efast/                      # Python package
  __init__.py               # Exports: efast.fusion(...)
  efast.py                  # Core fusion algorithm (fusion, upsample_array)
  s2_processing.py          # Sentinel-2 pre-processing helpers
  s2_cloud_native.py        # S2 L2A COG window reads via Element84 Earth Search (no auth)
  s3_processing.py          # Sentinel-3 compositing helpers (binning_s3 requires ESA SNAP)
  s3_openeo.py              # S3 SYN L2 AOI download via CDSE OpenEO (no SNAP required)
tests/
  test_temporal_weight.py
run_efast.py                # Original pipeline: full .SAFE + .SEN3 downloads, SNAP binning
run_cloud_native_efast.py   # Cloud-native pipeline: S2 COG range reads + S3 via OpenEO
pyproject.toml
requirements.txt
```

## Development Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]   # installs ruff for linting
```

Python **3.10–3.11** is required (`pyproject.toml` constraint). Python 3.12+ is not supported.

ESA SNAP is only required when running the **original** `run_efast.py` pipeline (the `binning_s3` step calls the SNAP GPT binary). The cloud-native pipeline (`run_cloud_native_efast.py`) uses CDSE OpenEO for S3 and does not need SNAP at all. SNAP v9/v10 can be downloaded from [ESA's site](https://step.esa.int/main/download/snap-download/).

## Running Tests

```bash
pytest tests/
```

## Linting

The project uses **ruff** (isort only):

```bash
ruff check --fix .
ruff format .
```

Run this before committing. Import order must follow the isort convention with `lines-between-types = 1` (one blank line between `import x` and `from x import y` blocks).

## Credentials

CDSE credentials are required for S3 downloads in both pipelines and are passed via environment variables:

```bash
export CDSE_USER=your@email.com
export CDSE_PASSWORD=yourpassword
```

- `run_efast.py` — used for OData S2/S3 granule downloads via `creodias-finder`
- `run_cloud_native_efast.py` — used for CDSE OpenEO authentication in `s3_openeo.py`; S2 uses the public Element84 Earth Search catalogue and requires no credentials

Never hardcode credentials in source files.

## Key Concepts

- **`fusion(pred_date, lr_dir, hr_dir, fusion_dir, product, ...)`** — the main entry point. Reads composited LR (`.tif`) and HR (`.tif` + `_DIST_CLOUD.tif`) files from disk, computes temporal and optical weights, interpolates and fuses, and writes output to `fusion_dir`.
- **LR files** are named `composite_YYYYMMDD.tif`; **HR files** encode the date at `split("_")[date_position]`.
- **`upsample_array`** — upsamples coarse arrays to HR resolution via `scipy.ndimage.zoom` (default) or `np.kron`.
- Weights combine a temporal Gaussian (σ = `sigma` days) with a ramp based on distance-to-cloud.

## Coding Conventions

- All public functions must have NumPy-style docstrings (Parameters / Returns / References).
- Reference specific paper sections/equations in inline comments where the math is non-trivial (e.g., `# Section 2.3 from [Senty2024]`).
- Avoid adding overview support by default (`add_overview=False`); the TODO in `efast.py` notes this should become `save_as_cog`.
- Use `pathlib.Path` for file paths throughout; avoid raw string concatenation.
- Do not introduce new top-level dependencies without updating both `pyproject.toml` and `requirements.txt`.
