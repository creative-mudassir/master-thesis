# Earth Data Cube Systems for Deep Learning Workflows: An Experimental Inter-Comparison

MSc Data Science thesis, HSE University — Mudassir Iftikhar
Supervisor: Prof. Ramon Antonio Rodriges Zalipynis

One end-to-end deep learning workflow (Sentinel-2 discovery → data cube → cloud masking and monthly
compositing → CNN / temporal Transformer inference → land-cover map) is implemented in four Earth
Data Cube systems and compared on runtime, development effort, capabilities, reproducibility and
map consistency.

| System | Status |
|---|---|
| Cloud-native (STAC + odc-stac + xarray + Zarr), reference | in progress |
| Google Earth Engine | planned |
| openEO (Copernicus Data Space Ecosystem) | planned |
| Open Data Cube | planned |

Study area: Moscow region (see `configs/aois.yaml`), Sentinel-2 L2A 2021, labels ESA WorldCover 2021.

## Quick start
```
conda env create -f environment.yml && conda activate edc && pip install torch
python scripts/01_check_access.py --gee-project <id>
python scripts/02_aoi_report.py
python scripts/03_build_cube.py --size-km 10
```
- Setup and accounts: `docs/SETUP_GUIDE.md`
- All steps: `docs/RUNBOOK.md`
- Fixed workflow rules: `configs/workflow.yaml`
- Decisions: `docs/decisions.md`

## Repository layout
```
configs/   workflow rules, study areas
src/edc/   shared code (grid, timing logger, cube, labels, dataset, models, metrics)
scripts/   01 … 09, run in order
docs/      guides, decision log, effort diary, capability matrix
outputs/   tables, figures, benchmark timings (maps and data are not in git)
```
