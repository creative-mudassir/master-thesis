# Runbook: A se Z tak kya chalana hai

Har step: **command → kya banega → kya check karna hai.**
Saari commands repo ke folder se, `conda activate edc` ke baad.

> **Chhote (10 km) test ke liye:** scripts 05–09 ka default `--size-km 50` hai. 10 km par test karte waqt
> har command mein `--size-km 10` lagayein, aur script 08 mein `--size-km 10 --train-size-km 10` dono.

> **Environment freeze (reproducibility):** jab environment theek chal jaaye, ek dafa:
> `conda env export > environment.lock.yml` aur `pip freeze > requirements.lock.txt`, phir dono commit karein.
> Kaggle par bhi `!pip freeze > models/kaggle_requirements.txt` chalayein.

---

## Step 1 — Access aur installation test (Day 1)
```
python scripts/01_check_access.py
```
- Banega: `outputs/tables/access_check.csv`
- Check: `pc`, `docker`, `gdalcubes` sab `OK` (`openeo_local` optional hai, SETUP_GUIDE step 7).
  Jo FAILED ho, uska error `docs/effort_diary.md` mein.

## Step 2 — AOI chunna (Day 1)
Pehle `configs/aois.yaml` mein 3-4 candidate AOIs likhein (**Russia se bahar**, D15; format file mein hai).
Grid ka CRS khud AOI ke UTM zone se banta hai (D17), kuch set karne ki zaroorat nahi.
```
python scripts/02_aoi_report.py
```
- Banega:
  - `outputs/figures/aoi_clear_dates.png` (har mahine kitni saaf tareekhein)
  - `outputs/figures/aoi_class_shares.png` (har AOI mein kaunsi classes kitni)
  - `outputs/tables/aoi_summary.csv`
  - `outputs/aois.geojson` → https://geojson.io par drag-drop karke boxes map par dekhein
- Kaise chunein:
  1. April-October ke **har mahine** mein kam se kam 1-2 clear dates hon.
  2. `class_entropy` zyada ho (classes balanced).
  3. Cropland, forest, built-up, water sab maujood hon.
  Chunne ke baad `configs/aois.yaml` mein `selected:` set karein.
- Months ka faisla naye AOI ke liye dobara karein (purana D2 Moscow ke liye tha, superseded):
  `python scripts/02b_snow_report.py` se har mahine ki clear dates aur snow dekhein, phir
  `workflow.yaml` ke `months` ka faisla `docs/decisions.md` mein likhein.

## Step 2b — Local Sentinel-2 archive (ek dafa, D16)
Saare systems **wahi local files** parhte hain, is liye timing internet par depend nahi karti.
```
python scripts/03a_download_archive.py --size-km 10     # pehle chhota test archive
python scripts/03a_download_archive.py                  # 50 km archive (raat ko); 10/25/50 km cubes sab isi se
```
- Banega: `data/archive/sentinel-2-l2a/<aoi>_<size>km/<scene id>/<band>.tif` (har band + SCL, COG),
  har scene ka STAC item `<scene id>/<scene id>.json`, aur `catalog.json` (local STAC catalogue) + `archive_info.json`.
  Har scene 50 km AOI box + 500 m tak kati hui, apne asli CRS aur pixels mein (resampling nahi).
- Scene filter (`max_scene_cloud_cover`) yahin ek dafa lagta hai, is liye har system ko bilkul wahi scenes milti hain.
- Download ka time `W0_download` stage mein alag record hota hai; systems ke muqable mein shamil nahi.
- Beech mein internet toot jaaye to wahi command dobara chalayein: jo scenes aa chuki hain woh skip hongi.

## Step 3 — Reference cube (Day 2)
```
python scripts/03_build_cube.py --size-km 10 --archive-km 10     # pehle chhota test
python scripts/04_make_labels.py --size-km 10
```
- Banega: `data/cubes/xarray/<aoi>_10km/m04.zarr ... m10.zarr`, `data/labels/<aoi>_10km.zarr`
  aur figures `cube_quicklook_xarray_<aoi>_10km.png`, `labels_<aoi>_10km.png`.
- `03_build_cube.py` sirf local archive parhta hai (default 50 km archive; 10 km test ke liye `--archive-km 10`).
- Check:
  - Quicklook mein har mahine ki image saaf ho (baadal na hon, kaale dhabbe kam hon).
  - "valid obs." row mein zyada tar pixels 0 se upar hon.
  - Label map asal area jaisa lage (Google Maps se compare karein).

Sab theek ho to poora AOI raat ko chalayein:
```
python scripts/03_build_cube.py --size-km 50
python scripts/04_make_labels.py --size-km 50
```
Run beech mein ruk jaaye to wahi command (**baghair** `--overwrite`) dobara chalayein: jo mahine aur parts ban
chuke hain woh skip ho jaate hain. Laptop charger par aur lid khuli rakhein (sleep se run ruk jaata hai).

## Step 4 — Dataset (Day 3)
```
python scripts/05_make_dataset.py --size-km 50
python scripts/07_train_rf.py --size-km 50
```
- Banega: `data/datasets/<aoi>_50km/` (lagbhag 3 GB), `outputs/figures/split_<aoi>_50km.png`,
  `models/<aoi>_50km/rf_seed0_meta.json` (RF ka validation score).
- Check: split figure mein train/val/test blocks bikhre hue hon; har split mein classes maujood hon
  (terminal mein "pixel classes" list mein zero na ho).

Jaldi test karna ho to yahi sab `--size-km 10` ke saath bhi chalta hai.

## Step 5 — Training (Day 4) — Kaggle GPU par
1. `data/datasets/<aoi>_50km` folder ko zip karein.
2. Kaggle → **Datasets → New Dataset** → zip upload karein, naam `edc-<aoi>-50km` → Create (private).
3. Kaggle → **New Notebook** → right panel:
   - Accelerator: **GPU**
   - Internet: **On**
   - Add Input: apna dataset.
4. Notebook cells:
   ```
   !git clone https://github.com/<user>/<repo>.git
   %cd <repo>
   !pip install -q onnx onnxruntime pyyaml
   !mkdir -p data/datasets && ln -s /kaggle/input/edc-<aoi>-50km data/datasets/<aoi>_50km
   !ls data/datasets/<aoi>_50km
   ```
   (Agar repo private hai to use public kar dein ya code zip karke dataset ki tarah upload karein.
   Agar `ls` mein files ek aur folder ke andar hon to symlink us andar wale folder ka banayein.)
   ```
   for m in ["cnn", "transformer"]:
       for s in [0, 1, 2]:
           !python scripts/06_train.py --model {m} --seed {s} --size-km 50
   ```
5. Khatam hone par `models/<aoi>_50km/` ko zip karke download karein
   (`!zip -r models.zip models` → right panel Output se download). Apne laptop par repo ke `models/` mein unzip karein.
- Check: har `*_meta.json` mein `val.macro_f1`, `best_epoch`, `onnx_max_abs_diff` (chhota hona chahiye, < 1e-3).
  `*_history.csv` se training curves ka figure baad mein banega.

## Step 6 — Pehla map (Day 5) — reference system end-to-end
```
python scripts/08_predict_map.py --system xarray --model cnn_seed0
python scripts/08_predict_map.py --system xarray --model transformer_seed0
python scripts/08_predict_map.py --system xarray --model rf_seed0
python scripts/09_evaluate.py
```
- Banega: `outputs/maps/xarray/*.tif` aur `outputs/tables/accuracy_<aoi>_50km.csv`.
- Check: GeoTIFF ko QGIS (free) mein kholein aur satellite basemap ke saath dekhein.
- **Yahan reference system mukammal hai.** Professor ko update bhejne ka achha waqt.

## Step 7 — ODC, rasdaman, gdalcubes, openEO local (Day 6-9)
Har system ka kaam ek hi hai: **local archive ki wahi files se, usi grid par, usi rules se monthly composites banao,
aur common format mein save karo.**
Common format: `data/cubes/<system>/<aoi>_50km/mXX.zarr`, variables B02…B12 + valid_count (CLAUDE.md, rule 2).
Grid: `data/cubes/xarray/<aoi>_<size>km/grid.json` (CRS = AOI ka UTM zone, bounds, shape).
Phir Step 6 wali commands `--system odc` / `--system rasdaman` / `--system gdalcubes` / `--system openeo_local` ke saath chalayein.

Claude Code se har system ka script likhwayein (SETUP_GUIDE section 8 mein prompt ki misaal hai):
- `scripts/odc_build_cube.py`:
  - Docker mein PostgreSQL + datacube; `datacube system init`, product add
  - archive ke `catalog.json` / scene item JSONs se index karein (files `data/archive/` se, volume mount)
  - `dc.load(...)` usi grid par, phir wahi `mask_and_median`
- `scripts/rasdaman_build_cube.py`:
  - local GeoTIFFs ingest karein (rasdaman ka import tool / WCS-T), phir WCPS query se SCL mask + monthly median
  - nateeja common grid par GeoTIFF/netCDF export karke Zarr
- `scripts/gdalcubes_build_cube.R` (+ chhota Python wrapper timing ke liye):
  - `create_image_collection()` local files se, `cube_view()` grid.json ke CRS/bounds/10 m/monthly ke saath,
    `filter_pixel` / mask se SCL, `reduce_time("median")`, export, phir Zarr
- `scripts/openeo_local_build_cube.py` (OPTIONAL):
  - openEO process graph (`load_stac` local `catalog.json`, SCL mask, `aggregate_temporal_period("month", "median")`,
    `resample_spatial` grid ke CRS par) openeo-processes-dask se local chalayein

**Baseline offset check:** processing baseline 04.00 (2022 se) wali scenes mein DN par +1000 offset hota hai.
Archive mein raw DN hain, is liye `harmonise_baseline_04` ka rule **har system** ko khud lagana hai
(har scene ke item JSON mein `s2:processing_baseline` hai). 2021 ka data aam taur par is se pehle ka hai, lekin har system
mein check karein aur `docs/decisions.md` mein likhein. Agar yeh alag ho to systems ke maps mein farq aa sakta hai.

Har system ke liye effort diary aur capability matrix zaroor bharein (installation = row I1, archive ingest = I2).
Agar koi system 2 din mein na chale to jahan tak pahunche likh dein aur aage barhein.

## Step 8 — Timing runs (Day 10)
Har system mein 10, 25 aur 50 km par 3-3 dafa (cube + inference), sab local archive se (network nahi). Claude Code se ek `scripts/run_benchmark.py` banwayein
jo yeh sab ek ke baad ek chalaye; raat bhar chalne dein. Sab kuch `outputs/benchmarks/timings.csv` mein aata hai.
- **Zaroori:** har repeat mein `03_build_cube.py --overwrite` lagayein. Is ke baghair bane hue mahine skip ho jaate hain,
  aur sirf search (W1) ka time record hota hai.
- Analysis mein sirf `status == ok` wali rows lein. `interrupted` (Ctrl+C) aur `error` rows timing mein shamil na karein;
  error rows ko capability/effort findings mein ginein.
- Memory ka column `peak_rss_mb` hai: stage ke dauran sab se zyada RAM.

## Step 9 — Analysis aur figures (Day 11-12)
- `python scripts/09_evaluate.py` → accuracy aur systems ke darmiyan agreement, disagreement maps.
- Claude Code se figures banwayein (sirf CSV files se):
  - runtime ke stacked bars
  - scaling curves
  - capability matrix heatmap
  - system effect vs model effect (RQ4)

## Step 10 — Likhna (har din + Day 12-14)
Har din `docs/effort_diary.md`. Methods ka draft `configs/workflow.yaml` + `docs/decisions.md` se ban jaata hai.
