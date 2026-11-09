"""Test of scripts/03a_download_archive.py (needs internet, ~10-20 min).

    python tests/run_archive_test.py --aoi austria_tulln

Downloads the 10 km test archive of the AOI and checks the safety nets:
1. watchdog: the first attempt of scene 2 freezes (--test-stall 2); it must be killed after
   --timeout-min and retried with success;
2. resume: the whole process tree is KILLED after >= 5 scenes are finished; the rerun must say
   "N already in the archive" and download only the rest;
3. catalogue: catalog.json lists every scene, every asset file exists, values are readable.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edc import config  # noqa: E402
from edc.archive import open_archive  # noqa: E402

KILL_AFTER = 5


def kill_tree(pid):
    parent = psutil.Process(pid)
    for p in parent.children(recursive=True) + [parent]:
        try:
            p.kill()
        except psutil.Error:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    a = ap.parse_args()
    adir = config.archive_dir(config.tag(a.aoi, 10))
    args = [sys.executable, "-u", str(ROOT / "scripts" / "03a_download_archive.py"), "--aoi", a.aoi, "--size-km", "10"]
    log1, log2 = adir.parent / f"{adir.name}_test_run1.log", adir.parent / f"{adir.name}_test_run2.log"
    done = lambda: len(list(adir.glob("*/*.json")))          # finished scenes = item JSON written

    with open(log1, "w") as f:
        proc = subprocess.Popen(args + ["--overwrite", "--test-stall", "2", "--timeout-min", "3"],
                                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        while done() < KILL_AFTER:
            if proc.poll() is not None:
                sys.exit(f"run 1 ended before it could be killed (exit {proc.returncode}), see {log1}")
            time.sleep(1)
        kill_tree(proc.pid)
        proc.wait()
    at_kill = done()
    t1 = log1.read_text()
    print(f"run 1 killed with {at_kill} finished scenes; catalog.json exists: {(adir / 'catalog.json').exists()}")
    timeouts = [line for line in t1.splitlines() if "TimeoutError" in line and "attempt 1 failed" in line]
    print("watchdog:", timeouts[0] if timeouts else "NO TimeoutError line")
    assert timeouts, "the frozen scene was not killed"
    assert not (adir / "catalog.json").exists(), "an unfinished archive must not have a catalogue"

    with open(log2, "w") as f:
        rc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    t2 = log2.read_text()
    status = [line for line in t2.splitlines() if "already in the archive" in line]
    print(f"run 2 exit {rc}; {status[0] if status else 'NO status line'}")
    assert rc == 0 and status and f"{at_kill} already in the archive" in status[0]

    items, info = open_archive(adir)
    assert len(items) == info["n_items"] == len(info["item_ids"])
    n_files = 0
    for it in items:
        for name, asset in it.assets.items():
            with rasterio.open(asset.href) as src:
                assert src.crs.to_epsg() == int(it.properties["proj:code"].split(":")[1])
                assert list(src.shape) == asset.extra_fields["proj:shape"]
                n_files += 1
    print(f"catalogue: {len(items)} scenes, {n_files} band files, all readable with matching CRS and shape")
    print("archive test passed")


if __name__ == "__main__":
    main()
