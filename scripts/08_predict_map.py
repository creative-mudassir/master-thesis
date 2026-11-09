"""Day 5: W6 deployment + W7 export: wall-to-wall land-cover map from a system's cube.

    python scripts/08_predict_map.py --system xarray --model cnn_seed0
    python scripts/08_predict_map.py --system xarray --model transformer_seed0
    python scripts/08_predict_map.py --system xarray --model rf_seed0

Every system writes its cube in the same format (data/cubes/<system>/<tag>/mXX.zarr),
so the same ONNX model file is applied to every system's cube.
Output: outputs/maps/<system>/<tag>_<model>.tif (uint8 class ids, 255 = no data)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_grid  # noqa: E402
from edc.bench import new_run_id, stage  # noqa: E402
from edc.cloudnative import open_cube  # noqa: E402
from edc.dataset import normalise  # noqa: E402


def predictor(model_path, kind):
    if kind == "rf":
        import joblib
        rf = joblib.load(model_path)
        return lambda x: rf.predict(x.reshape(len(x), -1)).astype(np.uint8)
    import onnxruntime as ort
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(model_path), providers=prov)
    return lambda x: sess.run(None, {"x": x.astype(np.float32)})[0].argmax(1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="xarray")
    ap.add_argument("--model", required=True, help="e.g. cnn_seed0, transformer_seed0, rf_seed0")
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=50)
    ap.add_argument("--train-size-km", type=float, default=50, help="AOI size the model was trained on")
    ap.add_argument("--window", type=int, default=1024)
    a = ap.parse_args()

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    tag = config.tag(aoi["name"], a.size_km)
    train_tag = config.tag(aoi["name"], a.train_size_km)
    st = json.load(open(config.dataset_dir(train_tag) / "stats.json"))
    mean, std, months, bands = np.array(st["mean"], np.float32), np.array(st["std"], np.float32), st["months"], st["bands"]
    kind = a.model.split("_")[0]
    mdir = config.models_dir(train_tag)
    mpath = mdir / (f"{a.model}.joblib" if kind == "rf" else f"{a.model}.onnx")
    predict = predictor(mpath, kind)

    cube, count = open_cube(config.cube_dir(a.system, tag), months, bands)
    T, B, H, W = cube.shape
    run = new_run_id()
    common = dict(system=a.system, aoi=aoi["name"], area_km=a.size_km, n_months=T, run_id=run)
    out = np.full((H, W), 255, np.uint8)
    win, pad = a.window, 32

    with stage(f"W6_inference_{kind}", **common) as info:
        for y0 in range(0, H, win):
            for x0 in range(0, W, win):
                y1, x1 = min(H, y0 + win), min(W, x0 + win)
                if kind == "cnn":
                    # read the window with a 32 px halo of real neighbours; where the AOI border
                    # cuts the halo, fill it by mirroring (reflect) so border pixels also get context
                    ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
                    X = cube.isel(y=slice(ya, yb), x=slice(xa, xb)).values
                    x = normalise(X, mean, std).reshape(T * B, yb - ya, xb - xa)
                    pt, pb = pad - (y0 - ya), pad - (yb - y1)
                    pl, pr = pad - (x0 - xa), pad - (xb - x1)
                    pb += (-(x.shape[1] + pt + pb)) % 4      # U-Net needs multiples of 4
                    pr += (-(x.shape[2] + pl + pr)) % 4
                    x = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), mode="reflect")
                    p = predict(x[None])[0][pad: pad + (y1 - y0), pad: pad + (x1 - x0)]
                else:               # pixel-wise models
                    X = cube.isel(y=slice(y0, y1), x=slice(x0, x1)).values       # (T,B,h,w)
                    h, w = X.shape[-2:]
                    x = normalise(X.reshape(T, B, -1).transpose(2, 0, 1), mean, std)  # (h*w,T,B)
                    p = np.concatenate([predict(x[i:i + 65536]) for i in range(0, len(x), 65536)]).reshape(h, w)
                nodata = (count.isel(y=slice(y0, y1), x=slice(x0, x1)).values == 0).all(0)
                p[nodata] = 255
                out[y0:y1, x0:x1] = p
            print(f"row {y0}/{H}")
        info["note"] = a.model

    xs, ys = cube.x.values, cube.y.values
    res = float(xs[1] - xs[0])
    transform = Affine(res, 0, xs[0] - res / 2, 0, -res, ys[0] + res / 2)
    tif = config.maps_dir(a.system) / f"{tag}_{a.model}.tif"
    with stage("W7_export_map", **common):
        with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                           crs=aoi_grid(aoi, a.size_km, wf).crs, transform=transform, nodata=255,
                           compress="deflate", tiled=True) as dst:
            dst.write(out, 1)
    print("saved", tif)


if __name__ == "__main__":
    main()
