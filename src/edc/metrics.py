"""Accuracy and map-agreement metrics."""
import numpy as np


def confusion(y_true, y_pred, k):
    ok = (y_true != 255) & (y_pred != 255)
    idx = y_true[ok].astype(np.int64) * k + y_pred[ok].astype(np.int64)
    return np.bincount(idx, minlength=k * k).reshape(k, k)


def scores(cm, class_names=None):
    cm = cm.astype(np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = cm.sum(1) > 0
    oa = tp.sum() / max(n, 1)
    pe = (cm.sum(0) * cm.sum(1)).sum() / max(n * n, 1)
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    out = {
        "n_pixels": int(n), "overall_accuracy": oa, "kappa": kappa,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * cm.sum(1)).sum() / max(n, 1)),
    }
    names = class_names or [str(i) for i in range(len(f1))]
    for i, nm in enumerate(names):
        if present[i]:
            out[f"f1_{nm}"] = float(f1[i])
    return out


def agreement(map_a, map_b, k):
    """Pixel agreement between two maps (e.g. two systems)."""
    cm = confusion(map_a, map_b, k)
    s = scores(cm)
    return {"agreement": s["overall_accuracy"], "disagree_pct": 100 * (1 - s["overall_accuracy"]),
            "kappa": s["kappa"], "n_pixels": s["n_pixels"]}
