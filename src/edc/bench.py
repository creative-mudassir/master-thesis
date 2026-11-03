"""Benchmark logger: every workflow stage in every system is timed with `stage()`.

Usage:
    from edc.bench import stage, new_run_id
    run_id = new_run_id()
    with stage("W1_discovery", system="cloudnative", aoi="kolomna", area_km=10,
               n_months=7, run_id=run_id):
        items = search(...)

One row per stage is appended to outputs/benchmarks/timings.csv.
"""
import contextlib
import csv
import datetime as dt
import os
import platform
import threading
import time
import uuid

import psutil

from .config import timings_csv

FIELDS = [
    "run_id", "timestamp", "system", "aoi", "area_km", "n_months", "stage",
    "seconds", "peak_rss_mb", "status", "host", "note",
]
# Analysis must use only rows with status == "ok".
# peak_rss_mb = highest memory of this process + its children during the stage (sampled every 0.25 s).


def new_run_id():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def log_row(row, csv_path=None):
    csv_path = csv_path or timings_csv()
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


class _PeakMemory:
    """Samples resident memory (this process + child processes) in a background thread."""

    def __init__(self, interval=0.25):
        self.interval, self.peak, self._stop = interval, 0.0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _rss(self):
        p = psutil.Process()
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**20

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = self._rss()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss())


@contextlib.contextmanager
def stage(name, *, system, aoi, area_km, n_months, run_id, note="", csv_path=None):
    t0 = time.perf_counter()
    status = "ok"
    info = {"note": note}
    mem = _PeakMemory()
    try:
        with mem:
            yield info      # caller may set info["note"] = "..." inside the block
    except KeyboardInterrupt:          # stopped by hand (Ctrl+C): never counts as a valid timing
        status = "interrupted"
        raise
    except BaseException as e:         # failed stages are logged too (they are findings)
        status = f"error: {type(e).__name__}"
        raise
    finally:
        sec = time.perf_counter() - t0
        log_row({
            "run_id": run_id,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "system": system, "aoi": aoi, "area_km": area_km, "n_months": n_months,
            "stage": name, "seconds": round(sec, 3),
            "peak_rss_mb": round(mem.peak, 1),
            "status": status, "host": platform.node(), "note": info.get("note", ""),
        }, csv_path)
        print(f"[bench] {system:12s} {name:24s} {sec:9.2f} s  peak {mem.peak:7.0f} MB  {status}")
