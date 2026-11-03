# Runbook: A se Z tak kya chalana hai

Har step: **command → kya banega → kya check karna hai.**
Saari commands repo ke folder se, `conda activate edc` ke baad.

> **Chhote (10 km) test ke liye:** scripts 05–09 ka default `--size-km 50` hai. 10 km par test karte waqt
> har command mein `--size-km 10` lagayein, aur script 08 mein `--size-km 10 --train-size-km 10` dono.

> **Environment freeze (reproducibility):** jab environment theek chal jaaye, ek dafa:
> `conda env export > environment.lock.yml` aur `pip freeze > requirements.lock.txt`, phir dono commit karein.
> Kaggle par bhi `!pip freeze > models/kaggle_requirements.txt` chalayein.

---

## Step 1 — Access test (Day 1)
```
python scripts/01_check_access.py --gee-project <your-project-id>
```
- Banega: `outputs/tables/access_check.csv`
- Check: teeno `OK`. Jo FAILED ho, uska error `docs/effort_diary.md` mein.

## Step 2 — AOI chunna (Day 1)
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
  Default `kolomna` hai. Chunne ke baad `configs/aois.yaml` mein `selected:` set karein.
- Winter (Nov-Mar) mein clear dates bohot kam hon to decision D2 (sirf Apr-Oct) sahi hai.

## Step 3 — Reference cube (Day 2)
```
python scripts/03_build_cube.py --size-km 10     # pehle chhota test (5-15 min)
python scripts/04_make_labels.py --size-km 10
```
- Banega: `data/cubes/cloudnative/kolomna_10km/m04.zarr ... m10.zarr`, `data/labels/kolomna_10km.zarr`
  aur figures `cube_quicklook_cloudnative_kolomna_10km.png`, `labels_kolomna_10km.png`.
- Check:
  - Quicklook mein har mahine ki image saaf ho (baadal na hon, kaale dhabbe kam hon).
  - "valid obs." row mein zyada tar pixels 0 se upar hon.
  - Label map asal area jaisa lage (Google Maps se compare karein).

Sab theek ho to poora AOI raat ko chalayein:
```
python scripts/03_build_cube.py --size-km 50
python scripts/04_make_labels.py --size-km 50
```
Beech mein internet toot jaaye to wahi command dobara chalayein: jo mahine ban chuke hain woh skip ho jaate hain.

## Step 4 — Dataset (Day 3)
```
python scripts/05_make_dataset.py --size-km 50
python scripts/07_train_rf.py --size-km 50
```
- Banega: `data/datasets/kolomna_50km/` (lagbhag 3 GB), `outputs/figures/split_kolomna_50km.png`,
  `models/kolomna_50km/rf_seed0_meta.json` (RF ka validation score).
- Check: split figure mein train/val/test blocks bikhre hue hon; har split mein classes maujood hon
  (terminal mein "pixel classes" list mein zero na ho).

Jaldi test karna ho to yahi sab `--size-km 10` ke saath bhi chalta hai.

## Step 5 — Training (Day 4) — Kaggle GPU par
1. `data/datasets/kolomna_50km` folder ko zip karein.
2. Kaggle → **Datasets → New Dataset** → zip upload karein, naam `edc-kolomna-50km` → Create (private).
3. Kaggle → **New Notebook** → right panel:
   - Accelerator: **GPU**
   - Internet: **On**
   - Add Input: apna dataset.
4. Notebook cells:
   ```
   !git clone https://github.com/<user>/<repo>.git
   %cd <repo>
   !pip install -q onnx onnxruntime pyyaml
   !mkdir -p data/datasets && ln -s /kaggle/input/edc-kolomna-50km data/datasets/kolomna_50km
   !ls data/datasets/kolomna_50km
   ```
   (Agar repo private hai to use public kar dein ya code zip karke dataset ki tarah upload karein.
   Agar `ls` mein files ek aur folder ke andar hon to symlink us andar wale folder ka banayein.)
   ```
   for m in ["cnn", "transformer"]:
       for s in [0, 1, 2]:
           !python scripts/06_train.py --model {m} --seed {s} --size-km 50
   ```
5. Khatam hone par `models/kolomna_50km/` ko zip karke download karein
   (`!zip -r models.zip models` → right panel Output se download). Apne laptop par repo ke `models/` mein unzip karein.
- Check: har `*_meta.json` mein `val.macro_f1`, `best_epoch`, `onnx_max_abs_diff` (chhota hona chahiye, < 1e-3).
  `*_history.csv` se training curves ka figure baad mein banega.

## Step 6 — Pehla map (Day 5) — reference system end-to-end
```
python scripts/08_predict_map.py --system cloudnative --model cnn_seed0
python scripts/08_predict_map.py --system cloudnative --model transformer_seed0
python scripts/08_predict_map.py --system cloudnative --model rf_seed0
python scripts/09_evaluate.py
```
- Banega: `outputs/maps/cloudnative/*.tif` aur `outputs/tables/accuracy_kolomna_50km.csv`.
- Check: GeoTIFF ko QGIS (free) mein kholein aur satellite basemap ke saath dekhein.
- **Yahan reference system mukammal hai.** Professor ko update bhejne ka achha waqt.

## Step 7 — GEE, openEO, ODC (Day 6-9)
Har system ka kaam ek hi hai: **usi grid par, usi rules se monthly composites banao, aur common format mein save karo.**
Common format: `data/cubes/<system>/kolomna_50km/mXX.zarr`, variables B02…B12 + valid_count (CLAUDE.md, rule 2).
Phir Step 6 wali commands `--system gee` / `--system openeo` / `--system odc` ke saath chalayein.

Claude Code se har system ka script likhwayein (SETUP_GUIDE section 8 mein prompt ki misaal hai):
- `scripts/gee_build_cube.py`:
  - `COPERNICUS/S2_SR_HARMONIZED`, SCL se mask, monthly median
  - export to Google Drive/GCS as GeoTIFF (crs EPSG:32637, `crsTransform` from grid.json), phir download karke Zarr
  - GEE ka apna Random Forest (`ee.Classifier.smileRandomForest`) bhi
- `scripts/openeo_build_cube.py`:
  - `load_collection("SENTINEL2_L2A")`, SCL mask, `aggregate_temporal_period("month", reducer="median")`
  - `resample_spatial(projection=32637, resolution=10)`, phir GeoTIFF/netCDF download karke Zarr
- `scripts/odc_build_cube.py`:
  - Docker mein PostgreSQL + datacube
  - `datacube system init`, product add, Planetary Computer STAC se index (`stac-to-dc`)
  - `dc.load(...)`, phir wahi `mask_and_median`

**Baseline offset check:** processing baseline 04.00 (2022 se) wali scenes mein DN par +1000 offset hota hai.
2021 ka data aam taur par is se pehle ka hai, lekin har system mein check karein:
- **cloud-native:** khud correct karta hai (`harmonise_baseline_04`)
- **GEE:** `S2_SR_HARMONIZED` already correct hai
- **openEO/CDSE aur ODC:** dekhein aur `docs/decisions.md` mein likhein ke offset kaise handle hua.
  Agar yeh alag ho to systems ke maps mein farq aa sakta hai.

Har system ke liye effort diary aur capability matrix zaroor bharein. Agar koi system 2 din mein na chale
to jahan tak pahunche likh dein aur aage barhein.

## Step 8 — Timing runs (Day 10)
Har system mein 10, 25 aur 50 km par 3-3 dafa (cube + inference). Claude Code se ek `scripts/run_benchmark.py` banwayein
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
