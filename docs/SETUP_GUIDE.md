# Setup Guide: accounts aur installation (step by step)

Is file mein har account aur har software ke steps hain. Order mein karein.
Har step ke end mein **"Check"** likha hai: woh chal jaaye to agla step karein.

> Websites ke screens waqt ke saath badal jaate hain. Agar koi button ka naam alag ho
> to milta-julta option chunein, ya Claude Code se poochein.

---

## 0. Pehle yeh dekh lein

| Cheez | Kam se kam | Behtar |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Free disk | 50 GB | 100 GB+ (local Sentinel-2 archive + cubes; Docker images bhi jagah lete hain) |
| Internet | Sirf **ek dafa** zaroori: local archive download karne ke liye (D16). Us ke baad saare systems local files se parhte hain | Download raat ko chalayein |
| OS | Windows 10/11 (WSL 2 + Docker ke saath), macOS ya Linux | — |

GPU zaroori nahi: training Kaggle ke free GPU par hogi (step 4).

**Systems sirf local hain (D15):** xarray + Zarr (reference), Open Data Cube, rasdaman, gdalcubes,
aur optional openEO local (openeo-processes-dask). Koi cloud account (GEE, CDSE) nahi chahiye.

---

## 1. Starter code ko apni GitHub repo mein daalein

1. Git install karein (Windows: https://git-scm.com/downloads/win , saari default settings theek hain).
   Yeh "Git Bash" bhi deta hai jo Claude Code ke liye behtar hai.
2. Apni repo clone karein:
   ```
   git clone https://github.com/<aap-ka-username>/<repo-name>.git
   cd <repo-name>
   ```
3. Is zip ki saari files (CLAUDE.md, configs/, src/, scripts/, docs/, environment.yml, .gitignore, README.md)
   repo ke folder mein copy kar dein.
4. Commit aur push:
   ```
   git add .
   git commit -m "Project skeleton: configs, reference pipeline, docs"
   git push
   ```
**Check:** GitHub par repo kholein, files nazar aa rahi hon.

---

## 2. Python environment (Miniforge)

1. Miniforge download karein: https://github.com/conda-forge/miniforge (Windows ke liye
   `Miniforge3-Windows-x86_64.exe`). Install karein ("Just me", default options).
2. Start menu se **Miniforge Prompt** kholein (macOS/Linux: normal terminal).
3. Repo ke folder mein jaa kar environment banayein (10-15 min lag sakte hain):
   ```
   cd C:\path\to\your\repo
   conda env create -f environment.yml
   conda activate edc
   ```
4. PyTorch install karein:
   - NVIDIA GPU **nahi** hai (zyada tar laptops): `pip install torch`
   - NVIDIA GPU hai: https://pytorch.org/get-started/locally/ par apna OS + "Pip" + CUDA chun kar jo command aaye woh chalayein.

**Check:**
```
python -c "import odc.stac, xarray, zarr, torch, onnxruntime; print('all ok')"
```
`all ok` aana chahiye. Har nayi terminal mein pehle `conda activate edc` likhna yaad rakhein.

---

## 3. Microsoft Planetary Computer (sirf download source)

**Account ki zaroorat nahi.** Sentinel-2 scenes aur WorldCover labels yahan se **ek dafa** download hote
hain (local archive, D16). Koi system timing ke dauran Planetary Computer se nahi parhta.

**Check:**
```
python scripts/01_check_access.py --only pc
```
Output mein `== pc: OK` aur July 2021 ke items ki ginti aani chahiye.

---

## 4. Kaggle (free GPU for training)

1. https://www.kaggle.com par sign up karein.
2. **Settings → Phone verification** karein. Is ke baghair GPU aur internet nahi milta.
3. Kaggle har hafte lagbhag 30 ghante free GPU deta hai (exact quota apne account mein dekhein).
   Hamare chhote models ke liye yeh kaafi hai.

Training ke steps `docs/RUNBOOK.md` → Step 5 mein hain.

**Check:** New Notebook → right panel → Accelerator mein "GPU" option nazar aaye.

---

## 5. Docker Desktop (Open Data Cube aur rasdaman ke liye)

1. https://www.docker.com/products/docker-desktop/ se install karein.
   Windows par WSL 2 maangega: installer jo kahe woh karein, phir restart.
2. **Check:** `docker run hello-world` → "Hello from Docker!" aaye, aur
   `python scripts/01_check_access.py --only docker` → `== docker: OK`.
3. Open Data Cube: PostgreSQL + `datacube` containers (setup `scripts/odc_*` ke saath likha jaayega).
4. rasdaman: official guide https://doc.rasdaman.org dekhein (Docker ya WSL Ubuntu packages).
   Jo install method chunein, woh aur us mein lagne wala waqt `docs/effort_diary.md` mein likhein
   (installation effort bhi ek result hai: capability matrix row I1).

Local archive ki files containers ke andar bhi chahiye: `data/archive/` ko volume ki tarah mount karein,
copy na karein, taake har system bilkul wahi files parhe.

---

## 6. R aur gdalcubes

1. R install karein: https://cran.r-project.org (Windows: "base" installer, default options).
2. R kholein aur chalayein: `install.packages("gdalcubes")`
3. **Check:** `python scripts/01_check_access.py --only gdalcubes` → `== gdalcubes: OK` aur version.
   (`Rscript` PATH par hona chahiye; na ho to R ka `bin` folder PATH mein daalein.)

---

## 7. openEO local (OPTIONAL)

Sirf agar waqt ho. Is ki dependencies `edc` env ko kharab kar sakti hain, is liye alag env:
```
conda create -n edc-openeo -c conda-forge python=3.11 -y
conda activate edc-openeo
pip install "openeo-processes-dask[implementations]"
```
**Check (isi env mein):** `python -c "import openeo_processes_dask; print('ok')"` → `ok`.

---

## 8. Claude Code

Claude Code ke liye **Pro, Max, Team ya Enterprise** plan chahiye (free plan mein shamil nahi).
Yeh sirf Anthropic ke supported countries se chalta hai.

1. Install:
   - Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`
   - macOS / Linux: `curl -fsSL https://claude.ai/install.sh | bash`
2. Nayi terminal kholein: `claude --version` → version number aaye.
3. Repo ke folder mein:
   ```
   cd C:\path\to\your\repo
   conda activate edc
   claude
   ```
   Pehli dafa browser mein login hoga.
4. Claude Code repo ki `CLAUDE.md` khud parhta hai, is liye use project ke rules pehle se pata honge.

### Claude Code se kaam kaise lein (achhe prompts)

- "Run `python scripts/03_build_cube.py --size-km 10`. If it fails, explain the error in simple words and fix it."
- "Explain what `src/edc/cloudnative.py` does, step by step, in simple English."
- "Write `scripts/odc_build_cube.py` that indexes the local archive in Open Data Cube and does W1-W3
  on exactly the grid from `edc.aoi.aoi_grid`, and saves the monthly composites in the common Zarr format
  (see CLAUDE.md rule 2). Time every stage with `edc.bench.stage`. Test on 10 km first."
- "Look at outputs/figures/cube_quicklook_*.png and tell me if the cloud masking looks right."

**Rules for yourself:**
- Ek prompt = ek kaam.
- Har script ke baad output (figure/CSV) khud dekhein.
- Jo cheez chal jaaye: `git add . && git commit -m "..."  && git push`.
- Paper ke numbers sirf apni CSV files se.
- Thesis/paper mein ek line likhein ke code likhne mein AI assistant (Claude Code) use hua
  (supervisor ne ijazat di hai; journal ki AI policy bhi check karein).

Official docs: https://code.claude.com/docs/en/setup
