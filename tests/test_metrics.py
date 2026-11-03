"""Synthetic test for macro_f1 vs macro_f1_main (decision D13).

    python tests/test_metrics.py        (or: pytest tests/)

Toy AOI with 1000 labelled pixels: Tree 600, Crop 395 (main classes) and Bare 5 (0.5%, rare).
The model is good on Tree and Crop but never finds Bare. One rare class with F1 = 0 pulls the
full macro_f1 down a lot, while macro_f1_main only averages the two main classes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from edc.metrics import confusion, main_classes, scores  # noqa: E402

NAMES = ["Tree", "Crop", "Bare"]
K = len(NAMES)


def toy_maps():
    y_true = np.array([0] * 600 + [1] * 395 + [2] * 5 + [255] * 20, np.uint8)  # 20 unlabelled pixels
    y_pred = y_true.copy()
    y_pred[:60] = 1             # 60 Tree pixels predicted as Crop
    y_pred[600:640] = 0         # 40 Crop pixels predicted as Tree
    y_pred[995:1000] = 0        # all 5 Bare pixels predicted as Tree
    return y_true, y_pred


def test_main_classes_threshold():
    y_true, _ = toy_maps()
    assert main_classes(y_true, K, 0.01) == [0, 1]           # Bare (0.5%) is not a main class
    assert main_classes(y_true, K, 0.001) == [0, 1, 2]       # lower threshold keeps it


def test_macro_f1_vs_main():
    y_true, y_pred = toy_maps()
    main = main_classes(y_true, K, 0.01)
    s = scores(confusion(y_true, y_pred, K), NAMES, main=main)

    ok = y_true != 255
    ref_all = f1_score(y_true[ok], y_pred[ok], labels=[0, 1, 2], average="macro")
    ref_main = f1_score(y_true[ok], y_pred[ok], labels=main, average="macro")
    assert abs(s["macro_f1"] - ref_all) < 1e-9
    assert abs(s["macro_f1_main"] - ref_main) < 1e-9
    assert s["f1_Bare"] == 0.0
    assert s["macro_f1_main"] > s["macro_f1"]
    return s


def test_without_main_no_extra_key():
    y_true, y_pred = toy_maps()
    assert "macro_f1_main" not in scores(confusion(y_true, y_pred, K), NAMES)


if __name__ == "__main__":
    test_main_classes_threshold()
    test_without_main_no_extra_key()
    s = test_macro_f1_vs_main()
    print("per-class F1:  " + ", ".join(f"{n} {s['f1_' + n]:.3f}" for n in NAMES))
    print(f"macro_f1      = {s['macro_f1']:.3f}   (all 3 classes, Bare F1 = 0 pulls it down)")
    print(f"macro_f1_main = {s['macro_f1_main']:.3f}   (main classes only: {NAMES[0]}, {NAMES[1]})")
    print("all tests passed")
