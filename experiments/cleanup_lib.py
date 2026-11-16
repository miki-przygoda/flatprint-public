"""Building blocks for the cleanup notebook: size-independent thumbnails and how far each one is from a reference."""

import numpy as np
from PIL import Image

SCORES = ("brightness", "shape", "spot")


def thumbnail(px, n=128) -> np.ndarray:
    """(h, w, 3) image -> float32 (n, n, 3) of block means, so images of any size can be compared."""
    x = np.asarray(px, dtype=np.float64)
    h, w = x.shape[:2]
    if h % n == 0 and w % n == 0:
        return x.reshape(n, h // n, n, w // n, 3).mean(axis=(1, 3)).astype(np.float32)
    return np.stack([np.asarray(Image.fromarray(x[..., c].astype(np.float32), "F").resize((n, n), Image.BOX))
                     for c in range(3)], axis=2).astype(np.float32)


def scores(thumb, ref) -> dict:
    """How a thumbnail differs from a reference thumbnail, in grey levels.

    brightness: difference of the overall mean level
    shape:      RMS difference once each has its own mean (per channel) removed (frames, shapes, gradients)
    spot:       largest single-point difference after the typical offset is taken away (small marks, text)
    """
    t, r = np.asarray(thumb, np.float64), np.asarray(ref, np.float64)
    d = t.mean(axis=2) - r.mean(axis=2)
    shape = (t - t.mean(axis=(0, 1))) - (r - r.mean(axis=(0, 1)))
    return {"brightness": float(abs(d.mean())),
            "shape": float(np.sqrt((shape ** 2).mean())),
            "spot": float(np.abs(d - np.median(d)).max())}


def calibrate(thumbs) -> np.ndarray:
    """Each reference thumbnail scored against the mean of the others; (m, 3) in SCORES order."""
    thumbs = np.asarray(thumbs, np.float64)
    total = thumbs.sum(axis=0)
    m = len(thumbs)
    return np.array([[scores(t, (total - t) / (m - 1))[k] for k in SCORES] for t in thumbs])
