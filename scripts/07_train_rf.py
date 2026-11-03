"""Day 3: Random Forest baseline (scikit-learn) on the pixel samples.

    python scripts/07_train_rf.py --size-km 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from edc import config  # noqa: E402
from edc.dataset import normalise  # noqa: E402
from edc.metrics import confusion, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    tag = config.tag(aoi["name"], a.size_km)
    ddir, mdir = config.dataset_dir(tag), config.models_dir(tag)
    st = json.load(open(ddir / "stats.json"))
    mean, std, names = np.array(st["mean"], np.float32), np.array(st["std"], np.float32), st["class_names"]
    hp = wf["training"]["rf"]

    X = normalise(np.load(ddir / "pixels_train_X.npy"), mean, std).reshape(-1, mean.size)
    Y = np.load(ddir / "pixels_train_Y.npy")
    rng = np.random.default_rng(a.seed)
    if len(X) > hp["max_train_samples"]:
        idx = rng.choice(len(X), hp["max_train_samples"], replace=False); X, Y = X[idx], Y[idx]
    rf = RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                min_samples_leaf=hp["min_samples_leaf"], n_jobs=-1,
                                class_weight="balanced_subsample", random_state=a.seed)
    t0 = time.time(); rf.fit(X, Y); sec = time.time() - t0
    Xv = normalise(np.load(ddir / "pixels_val_X.npy"), mean, std).reshape(-1, mean.size)
    Yv = np.load(ddir / "pixels_val_Y.npy")
    s = scores(confusion(Yv, rf.predict(Xv).astype(np.uint8), len(names)), names)
    joblib.dump(rf, mdir / f"rf_seed{a.seed}.joblib", compress=3)
    json.dump({"model": "rf", "seed": a.seed, "train_seconds": sec, "n_train": len(X), "val": s,
               "hyperparameters": hp}, open(mdir / f"rf_seed{a.seed}_meta.json", "w"), indent=1)
    print(f"RF trained in {sec:.0f}s; val OA {s['overall_accuracy']:.4f} macroF1 {s['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
