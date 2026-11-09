# Earth Data Cube Systems for Deep Learning Workflows: An Experimental Inter-Comparison

MSc Data Science thesis, HSE University — Mudassir Iftikhar
Supervisor: Prof. Ramon Antonio Rodriges Zalipynis

One end-to-end deep learning workflow (Sentinel-2 discovery → data cube → cloud masking and monthly
compositing → CNN / temporal Transformer inference → land-cover map) is implemented in several
**locally installed** Earth Data Cube systems and compared on runtime, development effort,
capabilities, reproducibility and map consistency.

| System (id) | Software | Status |
|---|---|---|
| `xarray` (reference) | xarray + dask + Zarr (odc-stac for loading) | in progress |
| `odc` | Open Data Cube (Docker: PostgreSQL + datacube) | planned |
| `rasdaman` | rasdaman (Docker) | planned |
| `gdalcubes` | gdalcubes (R package) | planned |
| `openeo_local` | openEO processes run locally (openeo-processes-dask) | optional |

**Same input for every system:** the Sentinel-2 L2A scenes are downloaded **once** into a local
archive (`data/archive/`), and all systems read the same local files, so timings do not depend on the
network (decision D16). The common grid uses the UTM zone of the study area (D17).

Study area: to be chosen (outside Russia, see `configs/aois.yaml`), Sentinel-2 L2A 2021,
labels ESA WorldCover 2021.

## Quick start
```
conda env create -f environment.yml && conda activate edc && pip install torch
python scripts/01_check_access.py
python scripts/02_aoi_report.py
python scripts/03_build_cube.py --size-km 10
```
- Setup: `docs/SETUP_GUIDE.md`
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
