"""Building blocks for the fingerprint notebook: residuals, averaging, comparison, spectra, display helpers."""

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

MODES = ("mean", "target", "highpass")


# ---- residuals and fingerprints -------------------------------------------------------------

def residuals(pixels, mode="mean", target=None, sigma=3.0) -> np.ndarray:
    """What is left of each image once its colour is taken away; float32 (n, h, w, 3).

    mean:     minus the image's own mean per channel
    target:   minus the RGB the prompt asked for
    highpass: minus a Gaussian-blurred copy (sigma in pixels), which also removes large-scale shading
    """
    x = np.asarray(pixels, dtype=np.float32)
    if mode == "mean":
        return x - x.mean(axis=(1, 2), keepdims=True)
    if mode == "target":
        if target is None:
            raise ValueError("mode 'target' needs target=(r, g, b)")
        return x - np.asarray(target, dtype=np.float32)
    if mode == "highpass":
        return x - gaussian_filter(x, sigma=(0, sigma, sigma, 0), mode="reflect")
    raise ValueError(f"unknown residual mode {mode!r}; use one of {MODES}")


def fingerprint_sum(pixels, idx, mode="mean", target=None, sigma=3.0, chunk=16):
    """Sum of the residuals of images `idx`, computed in chunks; returns (sum float64 (h, w, 3), n)."""
    idx = list(idx)
    total = np.zeros(pixels.shape[1:], dtype=np.float64)
    for s in range(0, len(idx), chunk):
        total += residuals(pixels[idx[s:s + chunk]], mode, target, sigma).sum(axis=0, dtype=np.float64)
    return total, len(idx)


def leave_one_out(total, n, r_i):
    """The fingerprint without image i: (sum - r_i) / (n - 1)."""
    return (total - r_i) / (n - 1)


def corr(a, b) -> float:
    """Pearson correlation of two arrays (means removed), over every element."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def slope(a, ref) -> float:
    """Least-squares slope of a on ref (means removed): how many copies of ref's pattern a holds. 1 = as strong
    as ref, 0 = none of it; times ref's RMS it is the pattern's strength in a's units. 0 if ref is flat."""
    a = np.asarray(a, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    a, r = a - a.mean(), r - r.mean()
    rr = r @ r
    return float(a @ r / rr) if rr > 0 else 0.0


def candidate_scales(width, max_width=2048, step=16, widths=None) -> list[float]:
    """Shrink factors to try for a picture `width` px wide: 1 (as it is) and width / w for every candidate original
    width w above width. Candidates are `widths` if given (e.g. the usual generator sizes), else every multiple of
    `step` up to max_width. Exact factors, because the pattern only comes back at (very nearly) the true one."""
    if widths is None:
        widths = range((width // step + 1) * step, max_width + 1, step)
    return [1.0] + [width / w for w in widths if w > width]


def best_of_k_null(pool, k, draws=20000, seed=0) -> tuple[float, float]:
    """Mean and sd of the best of k values drawn (with replacement) from `pool`: what a scan of k scales gives by
    chance when every value comes from the pool (e.g. real pictures' per-scale strengths)."""
    pool = np.asarray(pool, dtype=np.float64)
    best = np.random.default_rng(seed).choice(pool, size=(draws, k), replace=True).max(axis=1)
    return float(best.mean()), float(best.std())


def scale_scan(px, scales, sigs, ref, mode="highpass", sigma=3.0, period=16, max_width=2048) -> list[dict]:
    """Score a possibly resized picture at several guesses of how much it was shrunk. For each scale s, the
    picture is enlarged by 1/s (bicubic) back to its guessed original size, then residual -> fold -> strength
    (slope on ref) and score (best correlation with any of sigs). Scales whose enlarged width would exceed
    max_width are skipped; s = 1 (as it is) is always included. Returns one dict per scale tried."""
    img = Image.fromarray(np.asarray(px, dtype=np.uint8))
    w, h = img.size
    tried = sorted({float(s) for s in scales if 0 < s <= 1 and w / s <= max_width} | {1.0})
    out = []
    for s in tried:
        big = img if s == 1.0 else img.resize((round(w / s), round(h / s)), Image.Resampling.BICUBIC)
        a = np.asarray(big)
        a = a[:(a.shape[0] // period) * period, :(a.shape[1] // period) * period]
        f = fold(residuals(a[None], mode, None, sigma)[0], period)
        with np.errstate(invalid="ignore", divide="ignore"):
            cs = [corr(f, sg) for sg in sigs.values()]
        out.append({"scale": s, "strength": slope(f, ref),
                    "score": float(max((c for c in cs if np.isfinite(c)), default=0.0))})
    return out


def peak_sharpness(values) -> float:
    """How far the largest of a picture's per-scale strengths stands out from its others: (best - median of the
    rest) / (1.4826 * MAD of the rest), a robust z. A true resize gives one sharp spike, chance a flat curve.
    NaN with fewer than 4 values (no curve to judge)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 4:
        return float("nan")
    i = int(np.argmax(v))
    rest = np.delete(v, i)
    med = np.median(rest)
    mad = 1.4826 * np.median(np.abs(rest - med))
    if mad > 0:
        return float((v[i] - med) / mad)
    return 0.0 if v[i] == med else float("inf")     # a flat curve has no peak; a spike over a flat rest is infinite


def matched_z(value, k, refs, tolerance=3, leave_out=None) -> float:
    """z of `value` against the reference values whose k is within `tolerance` of this k (refs: (k, value)
    pairs, e.g. real pictures' best-of-scan strengths with their number of scales). leave_out removes one
    (k, value) pair, so a reference picture isn't compared with itself. NaN if fewer than 2 references match."""
    pool = [v for kk, v in refs if abs(kk - k) <= tolerance]
    if leave_out is not None and leave_out[0] is not None and abs(leave_out[0] - k) <= tolerance:
        pool.remove(leave_out[1])
    if len(pool) < 2:
        return float("nan")
    return float((value - np.mean(pool)) / np.std(pool, ddof=1))


def fit_logistic(X, y, balanced=True, l2=1e-3, iters=50) -> dict:
    """Logistic regression of y (0/1) on the columns of X (standardised inside), fitted by Newton's method with a
    small L2 penalty. balanced=True weights the two classes equally, so a probability means P(class 1) as if both
    classes were equally common, whatever their counts here."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.ones(len(y))
    if balanced:
        n1, n0 = y.sum(), len(y) - y.sum()
        w = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))
    beta = np.zeros(Z.shape[1])
    pen = np.r_[0.0, np.full(Z.shape[1] - 1, l2)]            # the intercept isn't penalised
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -35, 35)))
        grad = Z.T @ (w * (p - y)) + pen * beta * len(y)
        hess = (Z * (w * p * (1 - p))[:, None]).T @ Z + np.diag(pen * len(y))
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-10:
            break
    return {"beta": beta, "mu": mu, "sd": sd}


def predict_logistic(model, X) -> np.ndarray:
    """P(class 1) for each row of X under a fit_logistic model."""
    Z = np.c_[np.ones(len(X)), (np.asarray(X, dtype=np.float64) - model["mu"]) / model["sd"]]
    return 1 / (1 + np.exp(-np.clip(Z @ model["beta"], -35, 35)))


def cv_logistic(X, y, k=5, seed=0, **fit_kw):
    """Out-of-fold probabilities: rows split into k stratified folds, each fold predicted by a model fitted on the
    other k-1. Returns (probabilities, fold of each row)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        folds[idx] = np.arange(len(idx)) % k
    p = np.empty(len(y))
    for f in range(k):
        test = folds == f
        p[test] = predict_logistic(fit_logistic(np.asarray(X)[~test], y[~test], **fit_kw), np.asarray(X)[test])
    return p, folds


def grouped_cv_logistic(X, y, train, groups, k=5, seed=0, **fit_kw):
    """Out-of-fold probabilities by group: every group (e.g. an original picture and all its derived copies) goes
    to one of k folds, and every row of a fold, trained on or not, is predicted by a model fitted on the `train`
    rows of the other folds. So no prediction comes from a model that saw any row of its own group. Folds are
    balanced over groups that hold a positive and groups that don't. Returns (probabilities, fold of each row)."""
    X, y, train, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(train, bool), np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    pos = {g for g in np.unique(groups[(y == 1) & train])}
    fold_of = {}
    for subset in ([g for g in uniq if g in pos], [g for g in uniq if g not in pos]):
        for i, g in enumerate(rng.permutation(np.array(subset, dtype=object))):
            fold_of[g] = i % k
    folds = np.array([fold_of[g] for g in groups])
    p = np.empty(len(y))
    for f in range(k):
        fit_rows = train & (folds != f)
        p[folds == f] = predict_logistic(fit_logistic(X[fit_rows], y[fit_rows], **fit_kw), X[folds == f])
    return p, folds


def split_half_curve(res, sizes, repeats, rng) -> dict:
    """For each N: correlations between the means of two disjoint random sets of N residuals."""
    out = {}
    for n in sizes:
        if 2 * n > len(res):
            continue
        vals = []
        for _ in range(repeats):
            order = rng.permutation(len(res))
            vals.append(corr(res[order[:n]].mean(axis=0), res[order[n:2 * n]].mean(axis=0)))
        out[n] = vals
    return out


def fold(res, period=16) -> np.ndarray:
    """Average every period x period tile of a residual (h, w, 3) or a batch (n, h, w, 3) into one tile, so
    only what repeats with that period in pixels survives; images of different sizes give comparable tiles."""
    x = np.asarray(res, dtype=np.float64)
    h, w = x.shape[-3:-1]
    if h % period or w % period:
        raise ValueError(f"image {w}x{h} is not a multiple of the period {period}")
    lead = x.shape[:-3]
    x = x.reshape(*lead, h // period, period, w // period, period, 3)
    return x.mean(axis=(-5, -3)).astype(np.float32)


def masked_fold(res, mask, period=16) -> np.ndarray:
    """fold() of an (h, w, 3) residual counting only the pixels where mask (h, w) is true: each tile position is
    the average over the tiles where it is chosen (0 where it never is)."""
    x = np.asarray(res, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    h, w = x.shape[:2]
    if h % period or w % period:
        raise ValueError(f"image {w}x{h} is not a multiple of the period {period}")
    xs = (x * m[..., None]).reshape(h // period, period, w // period, period, 3).sum(axis=(0, 2))
    ms = m.reshape(h // period, period, w // period, period).sum(axis=(0, 2))
    return np.divide(xs, ms[..., None], out=np.zeros_like(xs), where=ms[..., None] > 0).astype(np.float32)


def region_masks(px, frac=0.3, sigma=4.0) -> dict:
    """Boolean (h, w) masks of the `frac` share of pixels that are darkest / brightest (grey level blurred by
    sigma) and smoothest / most detailed (local RMS of the grey high-pass, same blur). Picked by rank, so each
    mask covers exactly that share even where many pixels tie."""
    grey = np.asarray(px, dtype=np.float64).mean(axis=2)
    level = gaussian_filter(grey, sigma)
    detail = np.sqrt(gaussian_filter((grey - level) ** 2, sigma))
    k = int(round(frac * grey.size))

    def lowest(v):
        m = np.zeros(v.size, bool)
        m[np.argsort(v, axis=None, kind="stable")[:k]] = True
        return m.reshape(v.shape)

    return {"dark": lowest(level), "bright": lowest(-level), "smooth": lowest(detail), "detailed": lowest(-detail)}


def shift_corr(a, b) -> np.ndarray:
    """Correlation of tile a with tile b rolled by every cyclic shift (dy, dx); (p, p), [0, 0] = aligned."""
    p, q = np.asarray(b).shape[:2]
    return np.array([[corr(a, np.roll(b, (dy, dx), axis=(0, 1))) for dx in range(q)] for dy in range(p)])


def flat_noise(n, size, level, sigma, seed=0) -> np.ndarray:
    """n flat images (width, height) = size at one grey level plus independent Gaussian noise; uint8 (n, h, w, 3).
    A control with no structure of its own, for checking what the analysis itself produces."""
    w, h = size
    x = np.random.default_rng(seed).normal(level, sigma, (n, h, w, 3))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def roundtrip(px, ops_spec) -> np.ndarray:
    """An (h, w, 3) uint8 image through a flatprint op chain (e.g. "jpeg:90"), the same code `flatprint derive` uses."""
    from flatprint.ops import apply_ops, parse_ops
    out = apply_ops(png(px), parse_ops(ops_spec))
    return np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))


def percentile_of(value, values) -> float:
    """Percentage of `values` below `value` (ties count half)."""
    v = np.asarray(values, dtype=np.float64)
    return float(100 * ((v < value).sum() + 0.5 * (v == value).sum()) / len(v))


# ---- spectra ----------------------------------------------------------------------------------

def log_spectrum(img, hann=False) -> np.ndarray:
    """Centred log-magnitude spectrum of an (h, w, 3) array, channels averaged to grey, mean removed."""
    grey = np.asarray(img, dtype=np.float64).mean(axis=2)
    grey = grey - grey.mean()
    if hann:
        grey = grey * np.outer(np.hanning(grey.shape[0]), np.hanning(grey.shape[1]))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(grey))))


def local_contrast(spec, r=15) -> np.ndarray:
    """Spectrum minus its mean over a (2r+1)^2 box: how much each point stands out locally."""
    sat = np.pad(np.pad(spec, r, mode="edge"), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    k = 2 * r + 1
    box = (sat[k:, k:] - sat[:-k, k:] - sat[k:, :-k] + sat[:-k, :-k]) / (k * k)
    return spec - box


def _freqs(shape):
    h, w = shape
    fy, fx = np.mgrid[0:h, 0:w]
    return fy - h // 2, fx - w // 2


def find_peaks(spec, n, radius, skip_axes, rank_by="contrast") -> list[dict]:
    """Local maxima (5x5) of a centred spectrum, one of each mirrored pair, ranked; table rows."""
    h, w = spec.shape
    local_max = np.lib.stride_tricks.sliding_window_view(np.pad(spec, 2, mode="edge"), (5, 5)).max(axis=(2, 3))
    contrast = local_contrast(spec)
    fy, fx = _freqs(spec.shape)
    keep = (spec == local_max) & (fx**2 + fy**2 > radius**2) & ((fy > 0) | ((fy == 0) & (fx > 0)))
    if skip_axes:
        keep &= (fx != 0) & (fy != 0)
    score = contrast if rank_by == "contrast" else spec
    ys, xs = np.nonzero(keep)
    rows = []
    for rank, j in enumerate(np.argsort(score[ys, xs])[::-1][:n], 1):
        y, x = ys[j], xs[j]
        rows.append({"rank": rank, "fx": int(fx[y, x]), "fy": int(fy[y, x]),
                     "fx·8/W": round(float(fx[y, x]) * 8 / w, 3), "fy·8/H": round(float(fy[y, x]) * 8 / h, 3),
                     "period x (px)": round(w / abs(int(fx[y, x])), 2) if fx[y, x] else "∞",
                     "period y (px)": round(h / abs(int(fy[y, x])), 2) if fy[y, x] else "∞",
                     "log mag": round(float(spec[y, x]), 3), "contrast": round(float(contrast[y, x]), 3)})
    return rows


def lattice(w, h, div=16) -> list[tuple[int, int]]:
    """Frequencies (fx, fy) at every multiple of 1/div of the width and height, one of each mirrored
    pair, without the centre."""
    sx, sy = w // div, h // div
    pts = []
    for fy in range(-(h // 2), h - h // 2, sy):
        for fx in range(-(w // 2), w - w // 2, sx):
            if fy > 0 or (fy == 0 and fx > 0):
                pts.append((fx, fy))
    return pts


def lattice_strength(spec, points) -> np.ndarray:
    """Local contrast of a centred spectrum at each (fx, fy)."""
    c = local_contrast(spec)
    h, w = spec.shape
    return np.array([c[h // 2 + fy, w // 2 + fx] for fx, fy in points])


# ---- display ----------------------------------------------------------------------------------

def stretched(img, k):
    """Deviation from the image's own mean per channel, scaled so ±k std spans 0..255; returns (uint8, ±levels)."""
    dev = np.asarray(img, dtype=np.float64)
    dev = dev - dev.mean(axis=(0, 1))
    s = k * dev.std() or 1.0
    return np.clip(128 + 127 * dev / s, 0, 255).astype(np.uint8), s


def show_spectrum(spec, clip):
    lo, hi = np.percentile(spec, [100 - clip, clip])
    return (np.clip((spec - lo) / (hi - lo or 1.0), 0, 1) * 255).astype(np.uint8)


def with_guides(spec_u8, divs=(8, 16)):
    """Grey spectrum -> RGB with faint lines at every 1/div of the frequency range (1/8 orange, 1/16 blue)."""
    out = np.repeat(spec_u8[..., None], 3, axis=2).astype(np.float32)
    h, w = spec_u8.shape
    colours = {8: (255, 140, 0), 16: (0, 140, 255)}
    for div in sorted(divs, reverse=True):          # coarser lines drawn last, on top
        c = np.array(colours.get(div, (255, 0, 0)), dtype=np.float32)
        for k in range(-div // 2, div // 2 + 1):
            x, y = w // 2 + k * w // div, h // 2 + k * h // div
            if 0 <= x < w:
                out[:, x] = 0.45 * out[:, x] + 0.55 * c
            if 0 <= y < h:
                out[y, :] = 0.45 * out[y, :] + 0.55 * c
    return out.astype(np.uint8)


def zoom(img, x, y, size, scale):
    """A size x size crop at (x, y), each pixel blown up to a scale x scale square."""
    crop = np.asarray(img)[y:y + size, x:x + size]
    return crop.repeat(scale, axis=0).repeat(scale, axis=1)


def png(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(b, "PNG")
    return b.getvalue()
