"""Sweep of the residual's high-pass sigma: AUC of strength (as is) and score against the 36 social-media real pictures.

For each sigma the whole template is rebuilt as in controls.py section 7 (verification): residual (high-pass, that
sigma) -> 16-px fold -> five black signatures without the held-out blacks (same hold-out for every sigma: seed 0,
min(20, n // 2) per group, groups in controls.py's order), then every test picture is folded with the same sigma and
scored. strength = least-squares slope on the Z 1024 signature; score = best correlation with any signature.
sigma = 3 reproduces Table 2's strength column (held-out blacks 0.971, showcase 0.973, COCO 0.909, JPEG 85 0.923).

    uv run python experiments/sigma_sweep.py [--sigmas 1 2 3 5 8] [--out logs/results/sigma-sweep.csv]

Needs the social-media negatives, which are not in the public dataset.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import fingerprint_lib as fl  # noqa: E402
from flatprint.dataset import load_matrix, read_ids  # noqa: E402
from flatprint.store import load_manifest, resolve  # noqa: E402

DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
SIZES_SET, PERIOD, REF, MODE = "flat-black-sizes", 16, "Z 1024", "highpass"
HOLDOUT, HOLDOUT_SEED = 20, 0
TEST_SETS = {"Z scenes (showcase)": ["showcase", "adhoc"], "Z scenes (COCO)": ["z-scenes"],
             "Scenes, JPEG 85": ["showcase-q85"]}
SOCIAL = ["web-black-pinterest", "real-social", "real-social-hard"]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def black_groups(flagged):
    """controls.py load_data: the four sizes, then the earlier-run blacks ('Z old 1024'); clean rows only."""
    man = load_manifest(DATA)
    groups = {}
    for w, h in sorted({(r.width, r.height) for r in man if r.set == SIZES_SET}):
        m = load_matrix(DATA, SIZES_SET, size=(w, h))
        groups[f"Z {w}"] = m.pixels[[i for i, r in enumerate(m.records) if r.id not in flagged]]
    rgb = load_matrix(DATA, "flat-rgb")
    groups["Z old 1024"] = rgb.pixels[[i for i, (n, r) in enumerate(zip(rgb.names, rgb.records))
                                       if n == "black" and r.id not in flagged]]
    return groups


def folds(pixels, sigma):
    out = np.empty((len(pixels), PERIOD, PERIOD, 3), np.float32)
    for a in range(0, len(pixels), 16):
        out[a:a + 16] = fl.fold(fl.residuals(pixels[a:a + 16], MODE, None, sigma), PERIOD)
    return out


def file_pixels(path):
    """controls.py file_fold: top-left crop to a multiple of the period, no EXIF rotation."""
    px = np.asarray(Image.open(path).convert("RGB"))
    return px[: (px.shape[0] // PERIOD) * PERIOD, : (px.shape[1] // PERIOD) * PERIOD]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", nargs="+", type=float, default=[1, 2, 3, 5, 8])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "logs" / "results" / "sigma-sweep.csv")
    args = ap.parse_args()

    skip = set()
    for p in DATA.glob("*-flagged/flagged.md"):
        skip |= read_ids(p)
    blacks = black_groups(skip)
    rng = np.random.default_rng(HOLDOUT_SEED)
    held = {g: np.sort(rng.choice(len(px), min(HOLDOUT, len(px) // 2), replace=False)) for g, px in blacks.items()}
    print("held out per group:", {g: len(h) for g, h in held.items()}, flush=True)

    recs = [r for r in load_manifest(DATA) if r.id not in skip]
    if not any(r.set in SOCIAL for r in recs):
        sys.exit(f"no social-media negatives ({', '.join(SOCIAL)}) in {DATA}: they are third-party and not in the "
                 "public dataset, so this sweep only runs on the original store (its output: logs/results/sigma-sweep.csv)")
    tests = {name: [file_pixels(resolve(DATA, r)) for r in recs if r.set in sets] for name, sets in TEST_SETS.items()}
    social = [file_pixels(resolve(DATA, r)) for r in recs if r.set in SOCIAL]
    print("test pictures:", {k: len(v) for k, v in tests.items()}, "| social:", len(social), flush=True)

    rows = []
    for sigma in args.sigmas:
        f_black = {g: folds(px, sigma) for g, px in blacks.items()}
        sigs = {g: np.delete(F, held[g], axis=0).mean(axis=0) for g, F in f_black.items()}
        ref = sigs[REF]
        groups = {"Z black, held out": np.concatenate([F[held[g]] for g, F in f_black.items()]),
                  **{k: np.stack([folds(px[None], sigma)[0] for px in v]) for k, v in tests.items()}}
        f_social = np.stack([folds(px[None], sigma)[0] for px in social])

        def strength(F):
            return [fl.slope(f, ref) for f in F]

        def score(F):                                   # controls.py _score: an all-zero fold is "flat" -> 0
            out = []
            for f in F:
                with np.errstate(invalid="ignore", divide="ignore"):
                    cs = [fl.corr(f, s) for s in sigs.values()] if np.any(f) else [0.0]
                out.append(max(c if np.isfinite(c) else 0.0 for c in cs))
            return out

        neg_strength, neg_score = strength(f_social), score(f_social)
        for name, F in groups.items():
            rows.append({"sigma": sigma, "group": name, "n": len(F),
                         "AUC strength": round(auc(strength(F), neg_strength), 3),
                         "AUC score": round(auc(score(F), neg_score), 3)})
        print(f"sigma {sigma:g}: " + "  ".join(f"{r['group']} {r['AUC strength']:.3f}"
                                             for r in rows if r["sigma"] == sigma), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
