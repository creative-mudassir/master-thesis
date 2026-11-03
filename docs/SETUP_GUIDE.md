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
| Free disk | 50 GB | 100 GB |
| Internet | Stable. 50 km ka cube banane mein 10-20 GB data download hota hai | Raat ko chalayein |
| OS | Windows 10/11, macOS ya Linux | — |

GPU zaroori nahi: training Kaggle ke free GPU par hogi (step 6).

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
python -c "import odc.stac, xarray, torch, onnxruntime, ee, openeo; print('all ok')"
```
`all ok` aana chahiye. Har nayi terminal mein pehle `conda activate edc` likhna yaad rakhein.

---

## 3. Microsoft Planetary Computer (cloud-native system)

**Account ki zaroorat nahi.** Data free hai; `planetary-computer` package khud URLs sign kar deta hai.

**Check:**
```
python scripts/01_check_access.py --only pc
```
Output mein `== pc: OK` aur July 2021 ke items ki ginti aani chahiye.

---

## 4. Google Earth Engine (GEE)

1. Gmail account se login karein aur https://code.earthengine.google.com/register kholein.
2. **"Use with a Cloud project"** / non-commercial (unpaid) use chunein.
3. Use type: **Academia & Research** (aap student hain).
4. **Create a new Google Cloud project** chunein. Project ID likhein, masalan `edc-thesis-mudassir`.
   Yeh ID yaad rakhein, har jagah lagegi.
5. Form submit karein. Agar billing ya verification maange, to screen ki hidayat follow karein.
   Agar masla aaye (country, card, approval), screenshot lein aur `docs/effort_diary.md` mein likhein:
   yeh bhi ek finding hai (platform access barrier).
6. Terminal mein:
   ```
   earthengine authenticate
   ```
   Browser khulega, wahi Gmail chunein aur permission dein.

**Check:**
```
python scripts/01_check_access.py --only gee --gee-project edc-thesis-mudassir
```
`== gee: OK` aur images ki ginti aani chahiye.

Note: non-commercial projects par compute quota hoti hai. Cloud console mein Earth Engine ke
quota page par apni limit dekh kar `docs/effort_diary.md` mein note karein (yeh "cost/quota" table mein jaayegi).

---

## 5. Copernicus Data Space Ecosystem (openEO)

1. https://dataspace.copernicus.eu kholein → **Register** (ya "Login" → "Register").
2. Form bharein (naam, email, country, use = Research/Education). Email verify karein.
3. Terminal mein:
   ```
   python scripts/01_check_access.py --only openeo
   ```
   Pehli dafa terminal mein ek **link + code** aayega. Link browser mein kholein, apne CDSE account se
   login karein, code confirm karein. Script khud aage barh jaayegi.

**Check:** `== openeo: OK`.

Note: openEO free tier mein har mahine mehdood **credits** milte hain. Kitne credits hain aur kitne
kharch hue, yeh CDSE dashboard / openEO web editor (https://openeo.dataspace.copernicus.eu) mein dekh
kar note karein. Pehle hamesha 10 km area par test karein.

---

## 6. Kaggle (free GPU for training)

1. https://www.kaggle.com par sign up karein.
2. **Settings → Phone verification** karein. Is ke baghair GPU aur internet nahi milta.
3. Kaggle har hafte lagbhag 30 ghante free GPU deta hai (exact quota apne account mein dekhein).
   Hamare chhote models ke liye yeh kaafi hai.

Training ke steps `docs/RUNBOOK.md` → Step 6 mein hain.

**Check:** New Notebook → right panel → Accelerator mein "GPU" option nazar aaye.

---

## 7. Docker Desktop (Open Data Cube ke liye, Day 8 se pehle)

1. https://www.docker.com/products/docker-desktop/ se install karein.
   Windows par WSL 2 maangega: installer jo kahe woh karein, phir restart.
2. **Check:** `docker run hello-world` → "Hello from Docker!" aaye.

Abhi zaroori nahi; ODC wale din se pehle kar lein.

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
- "Write `scripts/gee_build_cube.py` that does W1-W3 in Google Earth Engine on exactly the grid from
  `edc.aoi.aoi_geobox`, exports the monthly composites, and saves them in the common Zarr format
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
