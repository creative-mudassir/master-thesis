"""Spatial block split, normalisation, CNN tiles and pixel samples."""
import numpy as np

SPLITS = ("train", "val", "test")


def block_grid(H, W, block_px):
    """Block id per pixel (row-major). Partial blocks at the edge are merged into the last one."""
    nby, nbx = max(1, H // block_px), max(1, W // block_px)
    by = np.minimum(np.arange(H) // block_px, nby - 1)
    bx = np.minimum(np.arange(W) // block_px, nbx - 1)
    return by[:, None] * nbx + bx[None, :], nby, nbx


def block_slices(b, nbx, block_px, H, W):
    r, c = divmod(b, nbx)
    nby = max(1, H // block_px)
    y0, x0 = r * block_px, c * block_px
    y1 = H if r == nby - 1 else y0 + block_px
    x1 = W if c == nbx - 1 else x0 + block_px
    return slice(y0, y1), slice(x0, x1)


def assign_blocks(labels, block_px, ratios, seed, n_classes, tries=500):
    """Random block -> split assignment; keeps the one where the rarest class
    (>=0.5% of AOI) is best represented in val and test."""
    H, W = labels.shape
    bid, nby, nbx = block_grid(H, W, block_px)
    nb = nby * nbx
    # class histogram per block
    hist = np.zeros((nb, n_classes), np.int64)
    ok = labels != 255
    np.add.at(hist, (bid[ok], labels[ok].astype(np.int64)), 1)
    frac = hist.sum(0) / max(1, hist.sum())
    important = np.where(frac >= 0.005)[0]

    n_tr = int(round(ratios[0] * nb))
    n_va = int(round(ratios[1] * nb))
    n_va = max(1, n_va)
    n_te = max(1, nb - n_tr - n_va)
    n_tr = nb - n_va - n_te
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    for _ in range(tries):
        perm = rng.permutation(nb)
        split = np.empty(nb, dtype="<U5")
        split[perm[:n_tr]] = "train"
        split[perm[n_tr:n_tr + n_va]] = "val"
        split[perm[n_tr + n_va:]] = "test"
        score = 1.0
        for s in ("val", "test", "train"):
            h = hist[split == s].sum(0)
            share = h[important] / np.maximum(1, hist.sum(0)[important])
            score = min(score, float(share.min()) if len(share) else 1.0)
        if score > best_score:
            best, best_score = split.copy(), score
    return {
        "block_px": block_px, "nby": nby, "nbx": nbx,
        "split": best.tolist(), "min_class_share": best_score,
    }


def split_mask(H, W, split_info, which, margin_px=0):
    """Bool mask of pixels belonging to split `which`, excluding block margins."""
    bp, nbx = split_info["block_px"], split_info["nbx"]
    m = np.zeros((H, W), bool)
    for b, s in enumerate(split_info["split"]):
        if s != which:
            continue
        ys, xs = block_slices(b, nbx, bp, H, W)
        m[ys.start + margin_px: ys.stop - margin_px, xs.start + margin_px: xs.stop - margin_px] = True
    return m


class RunningStats:
    """Per-channel mean/std over valid (>0) values, accumulated block by block."""

    def __init__(self, shape):
        self.n = np.zeros(shape, np.float64)
        self.s = np.zeros(shape, np.float64)
        self.ss = np.zeros(shape, np.float64)

    def add(self, x):  # x: (T, B, N) uint16
        v = x.astype(np.float64)
        ok = x > 0
        self.n += ok.sum(-1)
        self.s += (v * ok).sum(-1)
        self.ss += (v * v * ok).sum(-1)

    def result(self):
        n = np.maximum(self.n, 1)
        mean = self.s / n
        std = np.sqrt(np.maximum(self.ss / n - mean ** 2, 1e-6))
        return mean.astype(np.float32), std.astype(np.float32)


def normalise(x, mean, std):
    """x uint16 (..., T, B, [H, W]) -> float32, missing (0) -> 0 (= training mean)."""
    x = np.asarray(x)
    # supported layouts: (N,T,B) pixels, (T,B,H,W) or (N,T,B,H,W) tiles
    if x.shape[-2:] == mean.shape:           # (..., T, B)
        m, s = mean, std
    else:                                   # (..., T, B, H, W)
        m, s = mean[..., None, None], std[..., None, None]
    out = (x.astype(np.float32) - m) / s
    out[x == 0] = 0.0
    return out


def tiles_in_block(ys, xs, tile, margin):
    """Top-left corners of non-overlapping tiles fully inside the block interior."""
    y0, y1 = ys.start + margin, ys.stop - margin
    x0, x1 = xs.start + margin, xs.stop - margin
    out = []
    for y in range(y0, y1 - tile + 1, tile):
        for x in range(x0, x1 - tile + 1, tile):
            out.append((y, x))
    return out
