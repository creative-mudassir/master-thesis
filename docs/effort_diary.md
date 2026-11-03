# Effort diary

Fill this EVERY day (5 lines are enough). Setup hours and problems per system are
measured results of the study (RQ6), not just notes.

## Template
### YYYY-MM-DD — system: <cloudnative | gee | openeo | odc>
- Hours spent today:
- What I did:
- Problems / errors (exact message):
- Workaround / fix:
- Documentation used (link) and was it enough? (good / partial / missing)

---

### 2026-11-03 — system: setup
- Hours spent today: TODO (fill in actual hours)
- What I did: git repo created and pushed (github.com/creative-mudassir/master-thesis); conda env `edc`
  created from environment.yml + `pip install torch` (2.15.0 CPU); import check `all ok`;
  lock files environment.lock.yml + requirements.lock.txt committed.
- Problems / errors (exact message):
  1. `conda env create -f environment.yml` → `CondaToSNonInteractiveError: Terms of Service have not been
     accepted for the following channels` (repo.anaconda.com/pkgs/main, /r, /msys2), although
     environment.yml only lists conda-forge (Miniconda adds `defaults`).
  2. `conda env create ... --override-channels -c conda-forge` → `error: unrecognized arguments` (conda 25.11.1).
  3. `conda env export -n edc --file environment.lock.yml` → `EnvironmentExporterNotDetected: No environment
     exporter plugin found for filename 'environment.lock.yml'`.
  4. `pip freeze` wrote 185 of 221 lines as `pkg @ file:///home/conda/feedstock_root/...` (not reinstallable).
  5. pip downgraded pandas 3.1.0 (conda) → 2.3.3 while installing the pip packages (openeo/earthengine-api).
  6. `git push` once failed: `schannel: failed to receive handshake, SSL/TLS connection failed` (network).
- Workaround / fix:
  1. `CONDA_CHANNELS=conda-forge CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge conda env create -f environment.yml -y`
     (no ToS accepted, conda config unchanged). 3. `conda export -n edc --format=environment-yaml --file environment.lock.yml`.
  4. `pip list --format=freeze > requirements.lock.txt`. 5. accepted, `pip check` clean. 6. simple retry worked.
- Documentation used: SETUP_GUIDE step 2 — partial (does not mention the Anaconda ToS prompt or the
  conda 25 export change); conda error messages themselves were good.

### 2026-11-03 — system: gee
- Hours spent today: TODO (fill in actual hours, incl. registration)
- What I did: registered Earth Engine for **noncommercial use (Community Tier)**, Cloud project `edc-thesis`,
  eligibility valid until **2028-05-03**. Ran `python scripts/01_check_access.py --only gee --gee-project edc-thesis`
  → `gee: OK`, 12 images in COPERNICUS/S2_SR_HARMONIZED for July 2021 (10 km AOI), 34.3 s incl. first login
  (outputs/tables/access_check.csv).
- Problems / errors: none. First run needed a browser OAuth login (`ee.Authenticate()`, redirect to
  localhost:8085, scopes: earthengine, cloud-platform, drive, devstorage); token saved to
  ~/.config/earthengine/credentials.
- Workaround / fix: —
- Documentation used: SETUP_GUIDE step 4 — good. TODO: note the Community Tier compute quota from the
  Cloud console (needed for the cost/quota table).

### 2026-11-03 — system: openeo
- Hours spent today: TODO (fill in actual hours, incl. CDSE registration)
- What I did: `python scripts/01_check_access.py --only openeo` → first login via OIDC device code flow
  (link + user code, confirmed in browser, ~1 min); refresh token saved, later runs log in silently.
  After the fix: `openeo: OK`, SENTINEL2_L2A has 21 bands, all 10 workflow bands + SCL present (4.5 s).
- Problems / errors (exact message): first run → `TypeError: string indices must be integers, not 'str'`
  in our script (scripts/01_check_access.py): it expected band entries as dicts with "name", but CDSE
  returns `cube:dimensions.bands.values` as plain strings (as the openEO spec defines). Not a platform problem.
- Workaround / fix: script now reads the band names directly and checks that B02..B12 + SCL exist.
- Documentation used: SETUP_GUIDE step 5 — good. TODO: note the free-tier credits from the CDSE dashboard /
  openEO web editor (needed for the cost/quota table).

### 2026-11-03 — system: cloudnative (Planetary Computer)
- Hours spent today: TODO (no account needed, so only the check itself)
- What I did: `python scripts/01_check_access.py --only pc` → `pc: OK`, 12 S2 items in July 2021,
  256x256 B04 window read (mean DN 6838), 1 WorldCover item, 6.9 s.
- Problems / errors: first attempt hung for > 5 min without any output or error (stopped by hand).
  The two stuck python.exe processes could not be killed (`taskkill`: "There is no running instance of
  the task") → most likely stuck in a network call. A step-by-step test right after took 10 s
  (STAC root 200, search 8 s, signed COG HEAD 200, window read OK), and the rerun passed.
  Same session also had one `git push` TLS handshake failure → probably an unstable network, not PC.
  Harmless warning: `Cannot find gdalvrt.xsd (GDAL_DATA is not defined)` when the env's python is run
  without `conda activate edc`.
- Workaround / fix: rerun. For long cube builds: watch for silent stalls; D10 retries cover failed reads,
  but not reads that hang.
- Documentation used: SETUP_GUIDE step 3 — good.
- Later (Step 2, `02_aoi_report.py` + `02b_snow_report.py`): the Planetary Computer connection was reset
  twice: `APIError: HTTPSConnectionPool(host='planetarycomputer.microsoft.com', port=443): Max retries exceeded
  with url: /api/stac/v1 (Caused by ProtocolError('Connection aborted.', ConnectionResetError(10054, 'An existing
  connection was forcibly closed by the remote host')))`. A simple retry worked both times.
- Also: the first `02_aoi_report.py` run crashed without a Python traceback (exit 127) during the WorldCover load,
  when the env's python.exe was started directly without `conda activate edc` (GDAL_DATA not set, Git's
  mingw64 DLLs first on PATH). Fix: always run scripts in the activated env (`conda activate edc` or
  `conda run -n edc python ...`); the same script then ran fine.

### 2026-11-03 — system: cloudnative (Step 3: 10 km cube, scene-filter test, robust cube build)
- Hours spent today: TODO (fill in actual hours)
- What I did: built the 10 km cube (first with the 60% scene filter); tested the scene filter on September
  (60/80/100) and on all 7 months (60/80) with `scripts/03b_test_scene_filter.py` → D14 (filter 60 → 80).
  Rewrote `scripts/03_build_cube.py` to be robust (see "Workaround / fix"), rebuilt the 10 km cube with
  the 80% filter, and checked with `scripts/03c_check_parts.py` that building in parts gives the same cube.
- Problems / errors (exact message):
  1. **Process hang after the work was done:** the first `03_build_cube.py --size-km 10` printed `done -> ...`
     but the python process never exited (idle, 22 open sockets, PID 24156); it had to be killed with
     `taskkill /F`, so that shell reported exit code 1 although all months were written. Probably GDAL/dask
     network threads that keep the interpreter alive.
  2. **Network failures while reading COGs** (scene-filter test, 3 of 5 September runs failed; logged as
     `error` rows in `outputs/benchmarks/scene_filter_test.csv`):
     `RasterioIOError: CURL error: schannel: failed to receive handshake, SSL/TLS connection failed` and
     `RasterioIOError: Read failed. See previous exception for details.` (cut-off reads, "got 0 bytes").
     GDAL's own retries (`GDAL_HTTP_MAX_RETRY=10`) do not cover these errors. Same build took 5–47 s between tries.
  3. **Token expiry after a silent stall** (rebuild with the 80% filter, run 20261103-140547-6a0d83):
     the first attempt for September stalled for **13159 s (3.7 h)** without any error, and then failed with
     `RasterioIOError: HTTP response code: 403`. The SAS token in the URLs had `se=2026-11-03T09:54:51Z`;
     the attempt started at 09:05Z (14:05 local) and ended at 12:49Z (17:49 local), about 3 h after the token expired. Retry 2 (new tokens) took 12.4 s.
     Earlier the same day (07:22Z) a token check showed tokens last about 45 min, so a 50 km month that
     takes longer than that would fail the same way even without a stall.
     Note: the 3.7 h stall is NOT covered by the retries: a read that hangs never raises an error.
     Not fixed yet (would need a time limit per part).
  4. Months 6–8 took 49–89 s per month in this rebuild vs 9–16 s in the scene-filter test an hour earlier
     (same scenes, same code) → timing depends strongly on the network at the time of the run.
- Workaround / fix (`scripts/03_build_cube.py`, `src/edc/cloudnative.py`):
  1. forced exit `os._exit(code)` at the very end, only after all months and qa.json are written.
  2. each month is built in spatial parts (default 10 × 10 km); a failed part is retried up to 5 times,
     waiting 30, 60, 90 ... s; finished parts are listed in `mXX.parts.json`, so a rerun (without
     `--overwrite`) continues where it stopped; a month becomes `mXX.zarr` only when all parts are written.
  3. new download tokens before every part attempt (`resign()`: removes the old token from the URLs and
     empties the planetary_computer token cache, because `planetary_computer.sign()` skips URLs that are
     already signed). The 403 above was fixed exactly this way (attempt 2 worked).
  Check: real cube (1 part/month) and a test cube with 2.5 km parts (16 parts/month) are **identical, value
  for value**, to the one-go builds of the D14 test, all 7 months, all 11 variables
  (`outputs/tables/parts_check_kolomna_10km.csv`).
- Documentation used: Planetary Computer docs on SAS tokens — partial (no hint that `sign()` skips already
  signed URLs; found in the source of `planetary_computer/sas.py`).
