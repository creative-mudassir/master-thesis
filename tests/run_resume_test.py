"""Resume test for scripts/03_build_cube.py (needs internet, ~5 min).

    python tests/run_resume_test.py

1. builds September of the 10 km cube in 16 parts (2.5 km) into data/cubes/cloudnative_partstest/;
2. KILLS the whole process tree (like closing the terminal / a crash) after >= 6 parts are written;
3. reruns WITHOUT --overwrite and checks that it says "resuming, N/16" with the N parts that were done;
4. checks that the finished month is identical, value for value, to the one-go build of the D14 test.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cubes" / "cloudnative_partstest" / "kolomna_10km"
REF = ROOT / "data" / "cubes" / "cloudnative_scenefilter_test" / "kolomna_10km" / "m09_cc80.zarr"
ARGS = [sys.executable, "-u", str(ROOT / "scripts" / "03_build_cube.py"), "--size-km", "10",
        "--months", "9", "--part-km", "2.5", "--out-system", "cloudnative_partstest"]
KILL_AFTER = 6


def done_parts():
    try:
        return json.loads((OUT / "m09.parts.json").read_text())["done"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return []


def kill_tree(pid):
    parent = psutil.Process(pid)
    for p in parent.children(recursive=True) + [parent]:
        try:
            p.kill()
        except psutil.Error:
            pass


def main():
    log1, log2 = OUT.parent / "resume_run1.log", OUT.parent / "resume_run2.log"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(log1, "w") as f:
        proc = subprocess.Popen(ARGS + ["--overwrite"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        while len(done_parts()) < KILL_AFTER:
            if proc.poll() is not None:
                sys.exit(f"run 1 ended before it could be killed (exit {proc.returncode}), see {log1}")
            time.sleep(0.5)
        kill_tree(proc.pid)
        proc.wait()
    at_kill = sorted(done_parts())
    print(f"run 1 killed: {len(at_kill)}/16 parts recorded as done {at_kill}; "
          f"m09.zarr exists: {(OUT / 'm09.zarr').exists()}, m09.zarr.tmp exists: {(OUT / 'm09.zarr.tmp').exists()}")
    assert not (OUT / "m09.zarr").exists(), "an unfinished month must not look finished"

    with open(log2, "w") as f:
        rc = subprocess.run(ARGS, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log2.read_text()
    resumed = [line for line in text.splitlines() if "resuming" in line]
    built = [line for line in text.splitlines() if "part " in line and "written" in line]
    print(f"run 2 exit {rc}; {resumed[0] if resumed else 'NO resuming line'}; parts written in run 2: {len(built)}")
    assert rc == 0
    assert resumed and f"{len(at_kill)}/16" in resumed[0]
    assert len(built) == 16 - len(at_kill)

    a, r = xr.open_zarr(OUT / "m09.zarr"), xr.open_zarr(REF)
    diff = sum(int((a[v].values.astype(np.int64) != r[v].values.astype(np.int64)).sum()) for v in r.data_vars)
    print(f"resumed month vs one-go build: {diff} different values")
    assert diff == 0
    print("resume test passed")


if __name__ == "__main__":
    main()
