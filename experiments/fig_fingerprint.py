"""Figures: the black fingerprint (signature tile + spectrum) and the sample-size curve, for the report.

Same computation and display settings as the notebooks (fingerprint.py, resolutions.py): highpass residuals
(sigma 3), 16-px fold, flagged images left out, stretch +-4 std, Hann window, 1/8 + 1/16 guides, 99.9 % clip,
split-half curves with 10 repeats, seed 0. Optional extra sets (e.g. flux-black) get their own signature tile.

    uv run python experiments/fig_fingerprint.py [--extra flux-black:1024] [--out report/]
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fingerprint_lib as fl  # noqa: E402
from flatprint.dataset import load_matrix, read_ids  # noqa: E402
from flatprint.store import load_manifest  # noqa: E402

DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
SET = "flat-black-sizes"
FLAGGED = [DATA / "flat-black-sizes-flagged" / "flagged.md", DATA / "flat-rgb-flagged" / "flagged.md"]
SIGMA, PERIOD, K, CLIP, REPEATS = 3.0, 16, 4, 99.9, 10
NS = (1, 2, 4, 8, 16, 32, 64)
POOL, CLIP_POOLED = 8, 99.5
LABELS = {"flux-black": "FLUX"}  # display name of an --extra set in the tile title
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def clean_pixels(set_name, size, flagged):
    m = load_matrix(DATA, set_name, size=size)
    rows = [i for i, r in enumerate(m.records) if r.id not in flagged]
    return m.pixels, rows


def residual_stack(pixels, rows):
    out = np.empty((len(rows),) + pixels.shape[1:], dtype=np.float32)
    for a in range(0, len(rows), 16):
        out[a:a + 16] = fl.residuals(pixels[rows[a:a + 16]], "highpass", None, SIGMA)
    return out


def curves(res, rng, tile=False):
    """Split-half curves; control = the second mean shifted (full size: by 16..size-16 px, tile: cyclically by
    1..15 px in each direction)."""
    ns = [n for n in NS if 2 * n <= len(res)]
    aligned = fl.split_half_curve(res, ns, REPEATS, rng)
    shifted = {}
    for n in ns:
        vals = []
        for _ in range(REPEATS):
            o = rng.permutation(len(res))
            a, b = res[o[:n]].mean(0), res[o[n:2 * n]].mean(0)
            if tile:
                dy, dx = rng.integers(1, a.shape[0]), rng.integers(1, a.shape[1])
            else:
                dy, dx = rng.integers(16, a.shape[0] - 16), rng.integers(16, a.shape[1] - 16)
            vals.append(fl.corr(a, np.roll(b, (dy, dx), axis=(0, 1))))
        shifted[n] = vals
    return aligned, shifted


def tile_image(sig):
    img, level = fl.stretched(sig, K)
    return img, level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="append", default=[], help="SET:WIDTH, e.g. flux-black:1024 (square)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "report")
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent.parent / "outputs" / "figures")
    args = ap.parse_args()
    flagged = set().union(*(read_ids(p) for p in FLAGGED if p.exists()))
    sizes = sorted({(r.width, r.height) for r in load_manifest(DATA) if r.set == SET})
    plt.rcParams.update({"font.family": "serif", "font.size": 8})

    sigs, curve_data, fold_curve_data, spec_1024 = {}, {}, {}, None
    for size in sizes:
        px, rows = clean_pixels(SET, size, flagged)
        res = residual_stack(px, rows)
        sigs[f"Z {size[0]}"] = fl.fold(res, PERIOD).mean(axis=0)
        if size == (1024, 1024):
            spec_1024 = fl.log_spectrum(res.mean(axis=0), hann=True)
        curve_data[f"Z {size[0]} (n={len(rows)})"] = curves(res, np.random.default_rng(0))
        fold_curve_data[f"Z {size[0]} (n={len(rows)})"] = curves(fl.fold(res, PERIOD), np.random.default_rng(0), tile=True)
        print(f"{SET} {size[0]}x{size[1]}: {len(rows)} clean images")
        del res
    for spec in args.extra:
        name, w = spec.split(":")
        px, rows = clean_pixels(name, (int(w), int(w)), flagged)
        sigs[f"{name} {w}"] = fl.fold(residual_stack(px, rows), PERIOD).mean(axis=0)
        print(f"{name} {w}: {len(rows)} images")

    # F1: signature tiles (Z 1024 and extras, each on its own +-K std stretch, level printed) + Z 1024 spectrum
    tiles = ["Z 1024"] + [f"{s.split(':')[0]} {s.split(':')[1]}" for s in args.extra]
    fig, axes = plt.subplots(1, len(tiles) + 1, figsize=(3.3, 3.3 / (len(tiles) + 1) + 0.45),
                             gridspec_kw={"width_ratios": [1] * len(tiles) + [1]})
    for ax, name in zip(axes, tiles):
        img, level = tile_image(sigs[name])
        ax.imshow(img, interpolation="nearest")
        rms = float(np.sqrt(((sigs[name] - sigs[name].mean()) ** 2).mean()))
        set_name, w = name.rsplit(" ", 1)
        ax.set_title(f"{LABELS.get(set_name, set_name)} {w}\nRMS {rms:.3f}", fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
    # Local contrast (spectrum minus its 31x31 box mean), max-pooled POOLxPOOL so single-bin peaks survive the
    # shrink to column width. Ticks every 1/8 cycle per pixel (period 8 px); a 16-px pattern also fills the
    # half-way points.
    c = fl.local_contrast(spec_1024)
    h, w = c.shape
    pooled = c.reshape(h // POOL, POOL, w // POOL, POOL).max(axis=(1, 3))
    ax = axes[-1]
    ax.imshow(fl.show_spectrum(pooled, CLIP_POOLED), cmap="gray", interpolation="nearest",
              extent=(-0.5, 0.5, 0.5, -0.5))
    ticks = [-0.5, -0.25, 0, 0.25, 0.5]
    ax.set_xticks(ticks, ["-1/2", "-1/4", "0", "1/4", "1/2"], fontsize=5.5)
    ax.set_yticks(ticks, ["-1/2", "-1/4", "0", "1/4", "1/2"], fontsize=5.5)
    ax.set_xticks(np.arange(-0.5, 0.51, 1 / 16), minor=True)
    ax.set_yticks(np.arange(-0.5, 0.51, 1 / 16), minor=True)
    ax.tick_params(length=2, pad=1)
    ax.tick_params(which="minor", length=1.2)
    ax.set_xlabel("cycles / px", fontsize=6, labelpad=1)
    ax.set_title("Z 1024\nspectrum", fontsize=7)
    fig.tight_layout(pad=0.3)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig-fingerprint.pdf")
    fig.savefig(args.out / "fig-fingerprint.png", dpi=300)

    # F2: split-half curves, aligned (solid, mean +- sd) and shifted control (dashed); full-size residuals and
    # 16-px folds (what the detector uses)
    args.csv.mkdir(parents=True, exist_ok=True)
    for stem, data, what in (("fig-sample-size", curve_data, "full-size residual"),
                             ("fig-sample-size-fold", fold_curve_data, "16-px fold")):
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        rows_csv = []
        for colour, (name, (al, sh)) in zip(OKABE, data.items()):
            ns = list(al)
            ax.errorbar(ns, [np.mean(al[n]) for n in ns], [np.std(al[n]) for n in ns], marker="o", markersize=3,
                        capsize=2, linewidth=1, color=colour, label=name)
            ax.plot(ns, [np.mean(sh[n]) for n in ns], linestyle="--", linewidth=0.8, color=colour, alpha=0.6)
            rows_csv += [(name, n, np.mean(al[n]), np.std(al[n]), np.mean(sh[n])) for n in ns]
        ax.set_xscale("log", base=2)
        ax.set_xlabel("images per half ($N$)")
        ax.set_ylabel(f"split-half correlation\n({what})")
        ax.axhline(0, color="0.6", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False)
        fig.tight_layout(pad=0.3)
        fig.savefig(args.out / f"{stem}.pdf")
        fig.savefig(args.out / f"{stem}.png", dpi=300)
        with open(args.csv / f"{stem}.csv", "w", encoding="utf-8") as f:
            f.write("group,N,aligned mean,aligned sd,shifted mean\n")
            f.writelines(f"{g},{n},{a:.4f},{sd:.4f},{c:.4f}\n" for g, n, a, sd, c in rows_csv)

    names = list(sigs)
    print("signature correlations (aligned):")
    for i, a in enumerate(names):
        print("  " + a.ljust(16) + " ".join(f"{fl.corr(sigs[a], sigs[b]):+.3f}" for b in names))
    print(f"-> {args.out}/fig-fingerprint, fig-sample-size, fig-sample-size-fold (.pdf + .png); csv in {args.csv}")


if __name__ == "__main__":
    main()
