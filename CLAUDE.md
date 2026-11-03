# CLAUDE.md — project instructions for Claude Code

## Project
MSc thesis (HSE, Data Science), supervisor Prof. R. A. Rodriges Zalipynis.
**Experimental inter-comparison of Earth Data Cube (EDC) systems for a deep learning workflow.**
The SAME workflow (W1-W7) and the SAME trained models run in 4 systems; we compare
time, effort, capabilities, reproducibility and how different the resulting maps are.

Systems (folder name in data/cubes/ and outputs/maps/):
- `cloudnative` — STAC (Planetary Computer) + odc-stac + xarray/dask + Zarr  ← REFERENCE, done first
- `gee`         — Google Earth Engine (Python API, COPERNICUS/S2_SR_HARMONIZED)
- `openeo`      — openEO on Copernicus Data Space Ecosystem (openeo.dataspace.copernicus.eu)
- `odc`         — Open Data Cube (Docker: PostgreSQL + datacube, indexed from PC STAC)

Workflow: W1 discovery → W2 cube (10 bands, 10 m, EPSG:32637) → W3 SCL cloud mask + monthly median
→ W4 training data (ESA WorldCover 2021, spatial blocks) → W5 models (trained ONCE, outside systems)
→ W6 inference with the same ONNX file → W7 export GeoTIFF map.

## Hard rules
1. `configs/workflow.yaml` holds the fixed rules. NEVER change it silently. If a change is needed,
   ask the user and add an entry to `docs/decisions.md`.
2. Every system must produce its cube on EXACTLY the grid from `edc.aoi.aoi_geobox()` (same CRS,
   bounds, 10 m pixels) and save it in the common format:
   `data/cubes/<system>/<aoi>_<size>km/mXX.zarr` with variables B02..B12 (uint16 DN, 0 = no data)
   and `valid_count` (uint8), dims (y, x). Then `scripts/08_predict_map.py` works unchanged.
3. Time every workflow stage with `edc.bench.stage(...)` (writes outputs/benchmarks/timings.csv).
   Failed stages are findings, not something to hide.
4. Record platform-specific workarounds, problems and setup time in `docs/effort_diary.md`
   and native/workaround/impossible in `docs/capability_matrix.csv`.
5. Develop and test on `--size-km 10` first; only then run 50 km.
6. Never invent or "estimate" results. Numbers for the paper come only from CSV/JSON files in outputs/ and models/.
7. Explain non-obvious code to the user in simple words; he must be able to defend it in the viva.
8. Keep code simple and readable; one script per step in scripts/, shared code in src/edc/.

## Layout
configs/  workflow.yaml (fixed rules), aois.yaml (study areas)
src/edc/  config, aoi (grid), bench (timing), cloudnative, labels, dataset, models, metrics
scripts/  01_check_access … 09_evaluate (run in order, see docs/RUNBOOK.md)
docs/     SETUP_GUIDE.md, RUNBOOK.md, decisions.md, effort_diary.md, capability_matrix.csv
data/, models/ (binaries), outputs/maps/ are git-ignored.

## Environment
conda env `edc` (environment.yml) + PyTorch. Run scripts from the repo root: `python scripts/0X_....py`.
