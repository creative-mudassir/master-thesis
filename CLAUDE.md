# CLAUDE.md — project instructions for Claude Code

## Project
MSc thesis (HSE, Data Science), supervisor Prof. R. A. Rodriges Zalipynis.
**Experimental inter-comparison of Earth Data Cube (EDC) systems for a deep learning workflow.**
The SAME workflow (W1-W7) and the SAME trained models run in several LOCALLY INSTALLED systems;
we compare time, effort, capabilities, reproducibility and how different the resulting maps are.
Scope (supervisor, D15): no study area in Russia; no cloud platforms (no GEE, no CDSE openEO).

Systems (folder name in data/cubes/ and outputs/maps/):
- `xarray`       — xarray + dask + Zarr (odc-stac to load the local files)  ← REFERENCE, done first
- `odc`          — Open Data Cube (Docker: PostgreSQL + datacube), indexes the local archive
- `rasdaman`     — rasdaman (Docker), ingests the local archive
- `gdalcubes`    — gdalcubes (R package), image collection from the local archive
- `openeo_local` — OPTIONAL: openEO processes run locally (openeo-processes-dask)

Data (D16): Sentinel-2 L2A scenes are downloaded ONCE (Planetary Computer STAC) into the local archive
`data/archive/sentinel-2-l2a/<aoi>_<size>km/` by `scripts/03a_download_archive.py` (per scene: one COG per
band + SCL, cut to the AOI box + 500 m, native CRS/pixels, plus its STAC item; `catalog.json` = local STAC
catalogue, code in `src/edc/archive.py`). All systems read the SAME local files.

Workflow: W0 download archive (once, not compared) → W1 discovery → W2 cube (10 bands, 10 m, UTM zone of
the AOI, D17) → W3 SCL cloud mask + monthly median → W4 training data (ESA WorldCover 2021, spatial blocks)
→ W5 models (trained ONCE, outside systems) → W6 inference with the same ONNX file → W7 export GeoTIFF map.

## Hard rules
1. `configs/workflow.yaml` holds the fixed rules. NEVER change it silently. If a change is needed,
   ask the user and add an entry to `docs/decisions.md`.
2. Every system must produce its cube on EXACTLY the grid from `edc.aoi.aoi_grid()` (same CRS,
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
9. Systems read Sentinel-2 ONLY from the local archive (data/archive/). No system may read imagery from
   the network inside a timed stage; the one-time download is timed separately (W0_download).

## Layout
configs/  workflow.yaml (fixed rules), aois.yaml (study areas)
src/edc/  config, aoi (grid), bench (timing), cloudnative (STAC search + reference cube), labels, dataset, models, metrics
scripts/  01_check_access … 09_evaluate (run in order, see docs/RUNBOOK.md)
docs/     SETUP_GUIDE.md, RUNBOOK.md, decisions.md, effort_diary.md, capability_matrix.csv
data/ (incl. data/archive/), models/ (binaries), outputs/maps/ are git-ignored.

## Environment
conda env `edc` (environment.yml) + PyTorch. Run scripts from the repo root: `python scripts/0X_....py`.
