"""Check (not part of the workflow): does building a month in spatial parts change the result?

Compares, pixel by pixel, every variable (B02..B12, valid_count) of
  * the real cube  data/cubes/cloudnative/<tag>/mXX.zarr            (03_build_cube.py, default parts)
  * a test cube    data/cubes/cloudnative_partstest/<tag>/mXX.zarr  (03_build_cube.py --part-km 2.5)
against the one-go builds of the D14 test (data/cubes/cloudnative_scenefilter_test/<tag>/mXX_cc80.zarr).
Also recomputes the no-data and empty-water shares of the real cube, to compare with D14.

    python scripts/03c_check_parts.py --size-km 10

Output: outputs/tables/parts_check_<tag>.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from edc import config  # noqa: E402

WATER = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=10)
    ap.add_argument("--filter", type=int, default=80, help="scene filter of the one-go reference")
    a = ap.parse_args()
    wf = config.workflow()
    tag = config.tag(config.get_aoi(a.aoi)["name"], a.size_km)
    ref_dir = config.cube_dir("cloudnative_scenefilter_test", tag)
    cubes = {"cloudnative": config.cube_dir("cloudnative", tag),
             "cloudnative_partstest": config.cube_dir("cloudnative_partstest", tag)}
    water = xr.open_zarr(config.labels_path(tag))["label"].values == WATER
    variables = list(wf["bands"]) + ["valid_count"]

    rows = []
    for m in wf["months"]:
        ref = xr.open_zarr(ref_dir / f"m{m:02d}_cc{a.filter}.zarr")
        for name, d in cubes.items():
            p = d / f"m{m:02d}.zarr"
            if not p.exists():
                continue
            ds = xr.open_zarr(p)
            n_diff, max_abs = 0, 0
            for v in variables:
                x, r = ds[v].values.astype(np.int64), ref[v].values.astype(np.int64)
                assert x.shape == r.shape, (name, m, v, x.shape, r.shape)
                n_diff += int((x != r).sum())
                max_abs = max(max_abs, int(np.abs(x - r).max()))
            cnt = ds["valid_count"].values
            rows.append({"month": m, "cube": name, "reference": f"one-go cc{a.filter}",
                         "values_compared": len(variables) * cnt.size, "values_different": n_diff,
                         "max_abs_difference": max_abs, "identical": n_diff == 0,
                         "no_valid_pct": round(100 * float((cnt == 0).mean()), 2),
                         "water_empty_pct": round(100 * float((cnt[water] == 0).mean()), 2)})
    t = pd.DataFrame(rows)
    t.to_csv(config.tables_dir() / f"parts_check_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
