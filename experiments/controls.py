import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import functools
    import os
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    import cleanup_lib as cl
    import fingerprint_lib as fl
    from flatprint.dataset import load_matrix, read_ids
    from flatprint.store import load_manifest, resolve

    DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
    SIZES_SET = "flat-black-sizes"
    HALF_SET = "flat-black-half"
    WEB_SET = "web-black-pinterest"        # dark images downloaded from Pinterest (see its flagged list)
    BASELINE_SET = "phone-blank"         # phone photos with the lens covered (own photos, JPEG)
    FLAGGED_FILES = [DATA / "flat-black-sizes-flagged" / "flagged.md", DATA / "flat-rgb-flagged" / "flagged.md"]
    USED_SETS = {SIZES_SET, HALF_SET, BASELINE_SET, WEB_SET, "flat-rgb"}
    try:                         # `-- --all` (e.g. marimo export html NB -- --all): press every run button
        RUN_ALL = "all" in mo.cli_args()
    except Exception:            # imported as a module, outside marimo: read the command line itself
        RUN_ALL = "--all" in sys.argv
    REF = "Z 1024"                # the reference the shift search and the per-image scores use
    ROBUST_OPS = [("q85", "JPEG 85"), ("w736", "736 wide"), ("w736-q85", "736 wide + JPEG 85"),
                  ("w564-q85", "564 wide + JPEG 85")]   # derived-set suffix -> what was done (stage 1)
    COCO_TRAIN_OPS = ROBUST_OPS[1:]                    # the resized ones, made from z-scenes as training examples
    REAL_RESIZE_OPS = ROBUST_OPS[2:]                   # the social-media ones, made from the real phone photos
    # Okabe-Ito palette: stays distinct under the common colour-vision deficiencies. Heatmaps use magma
    # (sequential) and RdBu (diverging), which are colour-blind safe too.
    CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
          "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#999999", "black": "#000000"}
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[CB[k] for k in ("blue", "orange", "green", "vermillion",
                                                                         "purple", "sky", "grey", "black")])


@app.cell
def intro():
    mo.md(r"""
    # Controls for the fold signature

    The same residual → fold → signature steps as `resolutions.py` section 6, run on things that are **not**
    Z-Image blacks, and compared with the Z-Image black signatures (**Z 512 … Z 1280**, **Z old 1024**).

    1. **Z-Image reference signatures** (clean images only), and a **phone baseline** (1b)
    2. **Other colours**: the 12 `flat-rgb` colours at 1024²
    3. **Synthetic control**: flat grey + independent Gaussian noise, made here (PNG, nothing stored)
    4. **JPEG simulation**: the synthetic images through `jpeg:Q` (same code as `flatprint derive`, 4:2:0)
    5. **Half size**: `flat-black-half` = `flat-black-sizes` through `resize:0.5,png` (bicubic)
    6. **Any other set**: e.g. phone photos or downloaded images once imported with `flatprint import`
    7. **Verification**: the black fingerprint vs real, other generated and social media images
    8. **Regions**: which parts of an image (dark, bright, smooth, detailed) the match comes from
    9. **Strength**: how much of the pattern each image holds, on one scale (slope on the Z 1024 signature)

    Flagged ids (both flagged lists) are left out everywhere; for `flat-black-half` that means images whose
    parent was flagged. Correlations are Pearson over the whole tile (all three channels).
    """)
    return


@app.cell
def load_data():
    _flagged = set()
    for _p in FLAGGED_FILES:
        if _p.exists():
            _flagged |= read_ids(_p)
    _man = load_manifest(DATA)
    sources = {}
    with mo.status.spinner("loading the black sizes, flat-rgb and the half-size set (first time builds the cache)"):
        for _w, _h in sorted({(r.width, r.height) for r in _man if r.set == SIZES_SET}):
            _m = load_matrix(DATA, SIZES_SET, size=(_w, _h))
            _rows = [i for i, r in enumerate(_m.records) if r.id not in _flagged]
            sources[f"Z {_w}"] = SimpleNamespace(pixels=_m.pixels, size=(_w, _h), kind="Z-Image black", rows=_rows,
                                                 ids=[_m.records[i].id for i in _rows])
        _rgb = load_matrix(DATA, "flat-rgb")
        for _name in dict.fromkeys(_rgb.names):
            _rows = [i for i, (n, r) in enumerate(zip(_rgb.names, _rgb.records)) if n == _name and r.id not in _flagged]
            _key = "Z old 1024" if _name == "black" else f"colour: {_name}"
            sources[_key] = SimpleNamespace(pixels=_rgb.pixels, size=(_rgb.pixels.shape[2], _rgb.pixels.shape[1]),
                                            kind="Z-Image black" if _name == "black" else "Z-Image colour", rows=_rows,
                                            ids=[_rgb.records[i].id for i in _rows])
        _parent_size = {r.id: r.width for r in _man if r.set == SIZES_SET}
        for _w, _h in sorted({(r.width, r.height) for r in _man if r.set == HALF_SET}):
            _m = load_matrix(DATA, HALF_SET, size=(_w, _h))
            _src = {_parent_size[r.parent_id] for r in _m.records}
            _rows = [i for i, r in enumerate(_m.records) if r.parent_id not in _flagged]
            sources[f"half of {_src.pop()} → {_w}"] = SimpleNamespace(
                pixels=_m.pixels, size=(_w, _h), kind="half size", rows=_rows, ids=[_m.records[i].id for i in _rows])
    mo.ui.table([{"group": g, "kind": s.kind, "size": f"{s.size[0]}x{s.size[1]}", "clean images": len(s.rows)}
                 for g, s in sources.items()], selection=None, page_size=25, label="loaded groups")
    return (sources,)


@app.cell
def controls():
    mode_pick = mo.ui.radio(["mean", "highpass"], value="highpass", inline=True, label="residual")
    sigma_pick = mo.ui.slider(1.0, 10.0, value=3.0, step=0.5, label="highpass blur σ (px)", show_value=True)
    period_pick = mo.ui.dropdown({"8": 8, "16": 16, "32": 32}, value="16", label="fold period (px)")
    k_pick = mo.ui.slider(1, 10, value=4, label="stretch ±k std", show_value=True)
    mo.hstack([mode_pick, sigma_pick, period_pick, k_pick], justify="start")
    return k_pick, mode_pick, period_pick, sigma_pick


@app.cell
def helpers(k_pick, sources):
    @functools.lru_cache(maxsize=128)
    def group_folds(name, mode, sigma, period):
        """Every clean image of a loaded group, residual -> fold; float32 (n, period, period, 3)."""
        s = sources[name]
        out = np.empty((len(s.rows), period, period, 3), np.float32)
        for a in range(0, len(s.rows), 16):
            out[a:a + 16] = fl.fold(fl.residuals(s.pixels[s.rows[a:a + 16]], mode, None, sigma), period)
        return out

    @functools.lru_cache(maxsize=32)
    def synth_folds(n, size, level, noise, seed, ops, mode, sigma, period):
        """n synthetic flat-noise images (optionally through an op chain such as "jpeg:90"), folded."""
        out = np.empty((n, period, period, 3), np.float32)
        for i in range(n):
            px = fl.flat_noise(1, size, level, noise, seed=seed * 100_000 + i)[0]
            if ops:
                px = fl.roundtrip(px, ops)
            out[i] = fl.fold(fl.residuals(px[None], mode, None, sigma), period)[0]
        return out

    @functools.lru_cache(maxsize=1024)
    def file_fold(path, mode, sigma, period):
        """One stored image of any size, cropped from the top-left to a multiple of the period, residual -> fold;
        returns (fold, mean level). EXIF rotation is not applied: the pixels are used in the order they are stored."""
        px = np.asarray(Image.open(path).convert("RGB"))
        h, w = (px.shape[0] // period) * period, (px.shape[1] // period) * period
        return fl.fold(fl.residuals(px[None, :h, :w], mode, None, sigma), period)[0], float(px.mean(dtype=np.float64))

    def compare(label, n, sig, refs):
        """One table row: the signature against every reference (aligned), and the shift search against REF."""
        row = {"group": label, "n": n}
        for r, ref in refs.items():
            row[r] = round(fl.corr(sig, ref), 3)
        c = fl.shift_corr(sig, refs[REF])
        best = np.unravel_index(int(np.argmax(c)), c.shape)
        row[f"best shift vs {REF} (dy, dx)"] = str(tuple(int(v) for v in best))
        row["corr at best shift"] = round(float(c[best]), 3)
        row["median over shifts"] = round(float(np.median(c)), 3)
        return row

    def tiles(sigs):
        """Signatures side by side, each stretched to ±k std of itself."""
        out = []
        for name, sig in sigs.items():
            img, lv = fl.stretched(sig, k_pick.value)
            p = sig.shape[0]
            out.append(mo.vstack([mo.md(f"**{name}** · ±{lv:.3f}"),
                                  mo.image(fl.png(fl.zoom(img, 0, 0, p, max(1, 160 // p))), width=p * max(1, 160 // p))]))
        return mo.hstack(out, justify="start", wrap=True)

    def per_image_hist(series, title):
        """Histograms of per-image correlations, one per labelled series."""
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bins = np.linspace(-1, 1, 81)
        for label, vals in series.items():
            ax.hist(vals, bins=bins, alpha=0.5, label=f"{label} (n {len(vals)}, median {np.median(vals):.3f})")
        ax.set_xlabel(f"correlation of an image's fold with the {REF} signature")
        ax.set_ylabel("images")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    return compare, file_fold, group_folds, per_image_hist, synth_folds, tiles


@app.cell
def section_refs():
    mo.md(r"""
    ## 1 · Z-Image reference signatures

    Mean fold of the clean black images per size, plus the old 1024 blacks from `flat-rgb`.
    """)
    return


@app.cell
def refs_view(group_folds, mode_pick, period_pick, sigma_pick, sources, tiles):
    _args = (mode_pick.value, sigma_pick.value, period_pick.value)
    with mo.status.spinner("folding the Z-Image blacks"):
        ref_folds = {g: group_folds(g, *_args) for g, s in sources.items() if s.kind == "Z-Image black"}
    refs = {g: f.mean(axis=0) for g, f in ref_folds.items()}
    tiles(refs)
    return ref_folds, refs


@app.cell(hide_code=True)
def section_phone():
    mo.md(r"""
    ## 1b · Phone baseline (`phone-blank`)

    Photos taken on a phone with the lens covered, imported unchanged: JPEG, 4:2:0, EXIF orientation not applied.
    Two sizes (5712×4284 and 4032×3024), each its own group, plus **phone (all)**. Each image is cropped from the
    top-left to a multiple of the fold period. **phone (all)** is also an extra column in every table below.
    """)
    return


@app.cell(hide_code=True)
def phone_view(
    compare,
    file_fold,
    mode_pick,
    per_image_hist,
    period_pick,
    refs,
    sigma_pick,
    tiles,
):
    _p, _args = period_pick.value, (mode_pick.value, sigma_pick.value, period_pick.value)
    _recs = sorted((r for r in load_manifest(DATA) if r.set == BASELINE_SET), key=lambda r: (r.width, r.id))
    _by, _info = {}, []
    for _r in mo.status.progress_bar(_recs, title="folding the phone blanks"):
        _f, _lv = file_fold(str(resolve(DATA, _r)), *_args)
        _by.setdefault(f"phone {_r.width}x{_r.height}", []).append(_f)
        _info.append({"id": _r.id, "file": _r.extra.get("original_name", ""), "size": f"{_r.width}x{_r.height}",
                      "mean level": round(_lv, 2), f"corr with {REF}": round(fl.corr(_f, refs[REF]), 3)})
    phone_folds = {g: np.stack(f) for g, f in _by.items()}
    phone_folds["phone (all)"] = np.concatenate(list(phone_folds.values()))
    baselines = {g: f.mean(axis=0) for g, f in phone_folds.items()}
    all_refs = {**refs, "phone (all)": baselines["phone (all)"]}
    _A = phone_folds["phone (all)"].astype(np.float64)
    _asum, _an = _A.sum(axis=0), len(_A)
    _hist = per_image_hist({f"phone images vs {REF}": [fl.corr(f, refs[REF]) for f in _A],
                            "phone images vs phone (all) without them (LOO)": [fl.corr(f, (_asum - f) / (_an - 1)) for f in _A]},
                           "per-image folds of the phone blanks")
    mo.vstack([mo.ui.table([compare(g, len(phone_folds[g]), s, refs) for g, s in baselines.items()], selection=None,
                           label="phone signatures vs the Z-Image black references"),
               mo.hstack([_hist, tiles({**baselines, REF: refs[REF]})], justify="start"),
               mo.ui.table(_info, selection=None, page_size=15, label="per image")])
    return (all_refs,)


@app.cell
def section_colours():
    mo.md(r"""
    ## 2 · Other colours

    Each `flat-rgb` colour's signature against the black references, and colour against colour. The rows are
    ordered as the set was generated. `mean level` is the average pixel value of the clean images.
    """)
    return


@app.cell
def colours_view(
    all_refs,
    compare,
    group_folds,
    mode_pick,
    period_pick,
    refs,
    sigma_pick,
    sources,
    tiles,
):
    _args = (mode_pick.value, sigma_pick.value, period_pick.value)
    _names = [g for g, s in sources.items() if s.kind == "Z-Image colour"]
    with mo.status.spinner("folding the colours"):
        colour_sigs = {g: group_folds(g, *_args).mean(axis=0) for g in _names}
    _rows = []
    for _g in _names:
        _s = sources[_g]
        _row = compare(_g.removeprefix("colour: "), len(_s.rows), colour_sigs[_g], all_refs)
        _row["mean level"] = round(float(np.mean([_s.pixels[r].mean(dtype=np.float64) for r in _s.rows])), 1)
        _rows.append(_row)
    _all = {"black (Z old 1024)": refs["Z old 1024"], **{g.removeprefix("colour: "): v for g, v in colour_sigs.items()}}
    _labels = list(_all)
    _m = np.array([[fl.corr(_all[a], _all[b]) for b in _labels] for a in _labels])
    colour_fig, _ax = plt.subplots(figsize=(7.5, 6.5))
    _im = _ax.imshow(_m, cmap="RdBu_r", vmin=-1, vmax=1)
    _ax.set_xticks(range(len(_labels)), _labels, rotation=45, ha="right")
    _ax.set_yticks(range(len(_labels)), _labels)
    for (_i, _j), _v in np.ndenumerate(_m):
        _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=7)
    colour_fig.colorbar(_im, ax=_ax, shrink=0.8)
    _ax.set_title(f"{period_pick.value}-px signature correlation between colours (aligned)")
    colour_fig.tight_layout()
    mo.vstack([mo.ui.table(_rows, selection=None, page_size=15, label="colour signatures vs the black references"),
               mo.hstack([colour_fig, tiles(_all)], justify="start")])
    return


@app.cell
def section_synth():
    mo.md(r"""
    ## 3 · Synthetic control

    `n` flat images at one grey `level` plus independent Gaussian noise (std `noise`), seeded, kept in memory
    (PNG is lossless, so nothing is lost by not saving them). Nothing in them repeats with any period, so
    whatever their signature shows comes from the residual / fold steps or from chance.

    The clean Z-Image 1024² blacks have a median level of about 7.6 and a std of about 2.4 (`resolutions.py`).
    """)
    return


@app.cell
def synth_controls():
    n_pick = mo.ui.slider(16, 256, value=64, step=16, label="images", show_value=True)
    size_pick = mo.ui.dropdown({"512": 512, "768": 768, "1024": 1024, "1280": 1280}, value="1024", label="size (px)")
    level_pick = mo.ui.slider(0, 255, value=8, label="level", show_value=True)
    noise_pick = mo.ui.slider(0.5, 10.0, value=2.4, step=0.1, label="noise std", show_value=True)
    seed_pick = mo.ui.number(value=0, start=0, stop=10_000, label="seed")
    mo.hstack([n_pick, size_pick, level_pick, noise_pick, seed_pick], justify="start")
    return level_pick, n_pick, noise_pick, seed_pick, size_pick


@app.cell
def synth_view(
    all_refs,
    compare,
    level_pick,
    mode_pick,
    n_pick,
    noise_pick,
    per_image_hist,
    period_pick,
    ref_folds,
    refs,
    seed_pick,
    sigma_pick,
    size_pick,
    synth_folds,
    tiles,
):
    synth_args = (n_pick.value, (size_pick.value, size_pick.value), level_pick.value, noise_pick.value, seed_pick.value)
    _args = (mode_pick.value, sigma_pick.value, period_pick.value)
    with mo.status.spinner(f"making and folding {n_pick.value} synthetic images"):
        _F = synth_folds(*synth_args, None, *_args)
    _sig = _F.mean(axis=0)
    _Z = ref_folds[REF].astype(np.float64)
    _zsum, _zn = _Z.sum(axis=0), len(_Z)
    _ssum = _F.astype(np.float64).sum(axis=0)
    _hist = per_image_hist({
        f"{REF} images vs {REF} without them (LOO)": [fl.corr(f, (_zsum - f) / (_zn - 1)) for f in _Z],
        "synthetic images": [fl.corr(f, refs[REF]) for f in _F],
        "synthetic images vs synthetic signature (LOO)": [fl.corr(f, (_ssum - f) / (len(_F) - 1)) for f in _F],
    }, "per-image folds")
    mo.vstack([mo.ui.table([compare("synthetic (PNG)", len(_F), _sig, all_refs)], selection=None,
                           label="synthetic signature vs the black references"),
               mo.hstack([_hist, tiles({"synthetic": _sig, REF: refs[REF]})], justify="start")])
    return (synth_args,)


@app.cell
def section_jpeg():
    mo.md(r"""
    ## 4 · JPEG simulation

    The same synthetic images (section 3 settings) through `jpeg:Q`, decoded, then residual → fold. JPEG works
    on 8×8 blocks, and with 4:2:0 subsampling the colour is stored at half resolution, so one block of colour
    covers 16×16 pixels. The table has every reference, and the shift search against the reference.
    """)
    return


@app.cell
def jpeg_controls():
    q_pick = mo.ui.multiselect([str(q) for q in (100, 95, 90, 85, 75, 50, 30)], value=["95", "90", "75", "50"],
                               label="JPEG quality")
    jpeg_run = mo.ui.run_button(label="run JPEG simulation")
    mo.hstack([q_pick, jpeg_run], justify="start")
    return jpeg_run, q_pick


@app.cell
def jpeg_view(
    all_refs,
    compare,
    jpeg_run,
    mode_pick,
    per_image_hist,
    period_pick,
    q_pick,
    refs,
    sigma_pick,
    synth_args,
    synth_folds,
    tiles,
):
    mo.stop(not (jpeg_run.value or RUN_ALL), mo.md("Press **run JPEG simulation** (about 1 s per image per quality at 1024²)."))
    _args = (mode_pick.value, sigma_pick.value, period_pick.value)
    _qs = sorted((int(q) for q in q_pick.value), reverse=True)
    jpeg_folds = {}
    for _q in mo.status.progress_bar(_qs, title="JPEG simulation"):
        jpeg_folds[f"jpeg:{_q}"] = synth_folds(*synth_args, f"jpeg:{_q}", *_args)
    _sigs = {k: f.mean(axis=0) for k, f in jpeg_folds.items()}
    _rows = [compare(k, len(jpeg_folds[k]), s, all_refs) for k, s in _sigs.items()]
    _hist = per_image_hist({k: [fl.corr(f, refs[REF]) for f in F] for k, F in jpeg_folds.items()},
                           "per-image folds of the JPEG'd synthetic images")
    mo.vstack([mo.ui.table(_rows, selection=None, label="JPEG'd synthetic signatures vs the black references"),
               mo.hstack([_hist, tiles({**_sigs, REF: refs[REF]})], justify="start")])
    return


@app.cell
def section_half():
    mo.md(r"""
    ## 5 · Half size

    `flat-black-half` (each image of `flat-black-sizes` resized ×0.5, bicubic, saved as PNG). A pattern with a
    16-px period before resizing has an 8-px period after it, so the source's **32-px** signature shrunk 2× (2×2
    block mean) is what a 16-px tile of the result would line up with; that column is **vs source, shrunk**.
    The highpass σ is in pixels of whichever image it runs on, so it covers twice as much of the original
    picture after resizing.
    """)
    return


@app.cell
def half_view(
    all_refs,
    group_folds,
    mode_pick,
    period_pick,
    refs,
    sigma_pick,
    sources,
    tiles,
):
    _p = period_pick.value
    _halves = [g for g, s in sources.items() if s.kind == "half size"]
    _rows, _sigs = [], {}
    with mo.status.spinner("folding the half-size set"):
        for _g in _halves:
            _src = f"Z {_g.split()[2]}"
            _new = int(_g.split()[-1])
            _sig = group_folds(_g, mode_pick.value, sigma_pick.value, _p).mean(axis=0)
            _shrunk = cl.thumbnail(group_folds(_src, mode_pick.value, sigma_pick.value, 2 * _p).mean(axis=0), _p)
            _native = f"Z {_new}"
            _sigs[_g] = _sig
            _rows.append({"group": _g, "n": len(sources[_g].rows),
                          "vs Z at the new size (native)": round(fl.corr(_sig, refs[_native]), 3) if _native in refs else "",
                          "vs source, shrunk": round(fl.corr(_sig, _shrunk), 3),
                          "vs source (as is)": round(fl.corr(_sig, refs[_src]), 3),
                          f"vs {REF}": round(fl.corr(_sig, refs[REF]), 3),
                          "vs phone (all)": round(fl.corr(_sig, all_refs["phone (all)"]), 3)})
    mo.vstack([mo.ui.table(_rows, selection=None, label="half-size signatures"), tiles(_sigs)])
    return


@app.cell
def section_other():
    mo.md(r"""
    ## 6 · Any other set

    For sets brought in with `flatprint import` (phone photos, downloaded images) or any other stored set.
    Images come in at any size: each is cropped from the **top-left** to a multiple of the fold period, so the
    pixel grid (and any 8×8 / 16×16 codec block grid) keeps its position. EXIF rotation is **not** applied, for
    the same reason: the pixels are used in the order they are stored.
    """)
    return


@app.cell
def other_controls():
    _sets = sorted({r.set for r in load_manifest(DATA)} - USED_SETS)
    other_pick = mo.ui.dropdown(_sets, value=_sets[0] if _sets else None, label="set")
    other_max = mo.ui.slider(4, 400, value=100, step=4, label="max images", show_value=True)
    other_run = mo.ui.run_button(label="fold this set")
    mo.hstack([other_pick, other_max, other_run], justify="start")
    return other_max, other_pick, other_run


@app.cell
def other_view(
    all_refs,
    compare,
    file_fold,
    mode_pick,
    other_max,
    other_pick,
    other_run,
    per_image_hist,
    period_pick,
    refs,
    sigma_pick,
    tiles,
):
    mo.stop(not (other_run.value or RUN_ALL) or other_pick.value is None, mo.md("Pick a set and press **fold this set**."))
    _p = period_pick.value
    _recs = [r for r in load_manifest(DATA) if r.set == other_pick.value][:other_max.value]
    _F, _info = [], []
    for _r in mo.status.progress_bar(_recs, title=f"folding {other_pick.value}"):
        _f, _lv = file_fold(str(resolve(DATA, _r)), mode_pick.value, sigma_pick.value, _p)
        _F.append(_f)
        _info.append({"id": _r.id, "size": f"{_r.width}x{_r.height}", "format": Path(_r.file).suffix,
                      "mean level": round(_lv, 1),
                      f"corr with {REF}": round(fl.corr(_f, refs[REF]), 3)})
    _F = np.stack(_F)
    _sig = _F.mean(axis=0)
    _hist = per_image_hist({other_pick.value: [r[f"corr with {REF}"] for r in _info]}, "per-image folds")
    mo.vstack([mo.ui.table([compare(other_pick.value, len(_F), _sig, all_refs)], selection=None,
                           label="signature vs the black references"),
               mo.hstack([_hist, tiles({other_pick.value: _sig, REF: refs[REF]})], justify="start"),
               mo.ui.table(_info, selection=None, page_size=15, label="per image")])
    return


@app.cell(hide_code=True)
def section_verify():
    mo.md(r"""
    ## 7 · Verification: fingerprint vs real vs other generated vs social media

    The Z-Image fingerprint is rebuilt **without** a random set of held-out black images per group, then every
    test image gets one **score**: the highest aligned correlation of its fold with any of the five Z-Image black
    signatures (so an image of any size can be scored). Flagged images are left out everywhere.

    | kind | group | what it is |
    |---|---|---|
    | fingerprint test | **Z black, held out** | Z-Image blacks not used to build the fingerprint |
    | other generated | **Z colours** | `flat-rgb`, the 11 non-black colours (1024²) |
    | other generated | **Z black, half size** | `flat-black-half`: the black set resized ×0.5 (bicubic, PNG) |
    | other generated | **Z early flat tests** | `flat-pilot`: black, white and mid grey, earlier prompt wording, 768² and 1024² |
    | other generated | **Z scenes** | `showcase` + `adhoc`: photo-like content (1216×832, 1024²) |
    | other generated | **VAE decode only** | `vae-flat-rgb`: flat colours through Z-Image's VAE alone, no diffusion |
    | other generated | **VAE noisy flat input** | `vae-noise`: flat 8 / 64 / 128 grey + pixel noise (std 2.4), VAE alone |
    | other generated | **VAE random latents** | `vae-random-latent`: seeded Gaussian latents decoded by the VAE alone |
    | other generated | **VAE flat latent + noise 0.3** | `vae-flat-latent`: a flat 8 / 64 / 128 grey encoded once, its latent + noise (std 0.3) decoded (decoder only) |
    | other generated | **… 0.1 / 0.3 (CPU) / 0.6** | `vae-flat-latent-n01`, `-n03`, `-n06`: the latent-noise sweep, made on a Linux CPU host in float32 (the 0.3 above was the Mac's GPU in bfloat16); shown once generated |
    | real | **phone** | `phone-blank` (lens covered, JPEG) |
    | social media | **pinterest** | `web-black-pinterest` |
    | real | **real phone photos** | `real-phone`: everyday scenes, iPhone 11 HEIC (via iCloud) |
    | real | **real phone, dim** | `real-phone-dim`: dim rooms, iPhone 11 and 16 Plus (HEIC and JPEG) |
    | real | **real phone photos / dim, 736 or 564 wide + JPEG 85** | `real-phone-…`, `real-phone-dim-…`: those photos as they'd look posted (cross-validated by lineage with their originals) |
    | social media | **social media, looks real** | `real-social`: Pinterest photos that looked real |
    | unlabelled | **social media, maybe AI** | `ai-maybe-social`: Pinterest images that looked AI or were tagged AI-edited; scored, but on neither side of the AUCs |
    | social media | **social media, judged not AI** | `real-social-hard`: Pinterest images judged highly likely not AI, though hard to tell |
    | AI, other generator | **social media, judged likely AI** | `likely-ai-social`: Pinterest images judged highly likely AI (unknown generators, so not expected to carry Z-Image's pattern) |
    | other generated | **Z scenes (COCO)** | `z-scenes`: 300 Z-Image scenes from COCO captions (1024²) |
    | other generated | **Z scenes (COCO), 736 wide / 736 wide + JPEG 85 / 564 wide + JPEG 85** | `z-scenes-w736`, `-w736-q85`, `-w564-q85`: those scenes through the social-media steps, so the classifier sees resized Z-Image (cross-validated by lineage with their originals) |
    | FLUX (same VAE) | **FLUX black**, **FLUX scenes** | `flux-black`, `flux-scenes`: FLUX.1-schnell, which shares Z-Image's VAE; the same black prompt (1024², 768²) and the first 150 of the same captions. Scored, on neither side of the fit |
    | robustness test | **robust: scenes / held-out blacks, …** | `showcase-…`, `flat-black-sizes-…`: our own Z-Image pictures through a simulated social-media pipeline (JPEG 85; shrunk to 736 or 564 px wide, never enlarged; or both). For the blacks only copies of the held-out ones count |

    The **threshold** comes from the synthetic-noise control only (section 3 settings): the chosen percentile of
    the synthetic images' scores, so none of the test images set it. A score over it reads **matches Z-Image**.
    **AUC vs real** = how often an image of that group outscores a real or social-media image (1 = always, 0.5 =
    chance). An image whose residual is all zero has nothing to correlate and scores 0 (**flat**). The **per
    prompt** table splits each group by prompt (colour, grey level or scene), or by size for the black groups.
    """)
    return


@app.cell(hide_code=True)
def verify_controls():
    holdout_pick = mo.ui.slider(5, 60, value=20, step=5, label="held-out generated images per group", show_value=True)
    holdout_seed = mo.ui.number(value=0, start=0, stop=10_000, label="hold-out seed")
    pct_pick = mo.ui.slider(90.0, 100.0, value=99.0, step=0.5, label="threshold = percentile of synthetic scores",
                            show_value=True)
    verify_run = mo.ui.run_button(label="run verification")
    mo.hstack([holdout_pick, holdout_seed, pct_pick, verify_run], justify="start")
    return holdout_pick, holdout_seed, pct_pick, verify_run


@app.cell(hide_code=True)
def verify_view(
    file_fold,
    group_folds,
    holdout_pick,
    holdout_seed,
    mode_pick,
    pct_pick,
    period_pick,
    ref_folds,
    sigma_pick,
    sources,
    synth_args,
    synth_folds,
    verify_run,
):
    mo.stop(not (verify_run.value or RUN_ALL), mo.md("Press **run verification**."))
    _args = (mode_pick.value, sigma_pick.value, period_pick.value)
    _rng = np.random.default_rng(holdout_seed.value)
    # group -> (kind, [(image, size, fold)])
    _groups = {"Z black, held out": ("fingerprint test", []), "Z colours": ("other generated", []),
               "Z black, half size": ("other generated", []), "Z early flat tests": ("other generated", []),
               "Z scenes": ("other generated", []), "VAE decode only": ("other generated", []),
               "VAE noisy flat input": ("other generated", []), "VAE random latents": ("other generated", []),
               "VAE flat latent + noise 0.1": ("other generated", []),
               "VAE flat latent + noise 0.3": ("other generated", []),
               "VAE flat latent + noise 0.3 (CPU)": ("other generated", []),
               "VAE flat latent + noise 0.6": ("other generated", []),
               "phone": ("real", []), "pinterest": ("social media", []),
               **{f"real phone photos, {_name}": ("real", []) for _what, _name in REAL_RESIZE_OPS},
               **{f"real phone, dim, {_name}": ("real", []) for _what, _name in REAL_RESIZE_OPS},
               "real phone photos": ("real", []), "real phone, dim": ("real", []),
               "social media, looks real": ("social media", []), "social media, judged not AI": ("social media", []),
               "social media, maybe AI": ("unlabelled", []), "social media, judged likely AI": ("AI, other generator", []),
               "Z scenes (COCO)": ("other generated", []),
               **{f"Z scenes (COCO), {_name}": ("other generated", []) for _what, _name in COCO_TRAIN_OPS},
               "FLUX black": ("FLUX (same VAE)", []), "FLUX scenes": ("FLUX (same VAE)", [])}
    for _what, _name in ROBUST_OPS:                     # stage 1: the social-media pipeline simulated on our own
        _groups[f"robust: scenes, {_name}"] = ("robustness test", [])
        _groups[f"robust: held-out blacks, {_name}"] = ("robustness test", [])
    verify_sigs = {}
    _held_ids = set()
    for _g, _F in ref_folds.items():
        _held = np.sort(_rng.choice(len(_F), min(holdout_pick.value, len(_F) // 2), replace=False))
        _held_ids |= {sources[_g].ids[_i] for _i in _held}
        verify_sigs[_g] = np.delete(_F, _held, axis=0).mean(axis=0)
        _w, _h = sources[_g].size
        _groups["Z black, held out"][1].extend((sources[_g].ids[_i], f"{_g} #{int(_i)}", _g, f"{_w}x{_h}", _F[_i])
                                               for _i in _held)
    for _g, _s in sources.items():                      # loaded groups: clean rows only
        _to = {"Z-Image colour": "Z colours", "half size": "Z black, half size"}.get(_s.kind)
        if _to:
            _groups[_to][1].extend((_s.ids[_k], f"{_g} #{_k}", _g.removeprefix("colour: "), f"{_s.size[0]}x{_s.size[1]}", _f)
                                   for _k, _f in enumerate(group_folds(_g, *_args)))
    _skip = set()
    for _fp in [*FLAGGED_FILES, *DATA.glob("*-flagged/flagged.md")]:
        if _fp.exists():
            _skip |= read_ids(_fp)
    _by_set = {"flat-pilot": "Z early flat tests", "showcase": "Z scenes", "adhoc": "Z scenes",
               "vae-flat-rgb": "VAE decode only", "vae-noise": "VAE noisy flat input",
               "vae-random-latent": "VAE random latents", "vae-flat-latent": "VAE flat latent + noise 0.3",
               "vae-flat-latent-n01": "VAE flat latent + noise 0.1", "vae-flat-latent-n06": "VAE flat latent + noise 0.6",
               "vae-flat-latent-n03": "VAE flat latent + noise 0.3 (CPU)",
               BASELINE_SET: "phone", WEB_SET: "pinterest", "real-phone": "real phone photos",
               "real-phone-dim": "real phone, dim", "real-social": "social media, looks real",
               "ai-maybe-social": "social media, maybe AI", "real-social-hard": "social media, judged not AI",
               "likely-ai-social": "social media, judged likely AI", "z-scenes": "Z scenes (COCO)",
               **{f"z-scenes-{_what}": f"Z scenes (COCO), {_name}" for _what, _name in COCO_TRAIN_OPS},
               **{f"real-phone-{_what}": f"real phone photos, {_name}" for _what, _name in REAL_RESIZE_OPS},
               **{f"real-phone-dim-{_what}": f"real phone, dim, {_name}" for _what, _name in REAL_RESIZE_OPS},
               "flux-black": "FLUX black", "flux-scenes": "FLUX scenes"}
    for _what, _name in ROBUST_OPS:
        _by_set[f"showcase-{_what}"] = f"robust: scenes, {_name}"
        _by_set[f"{SIZES_SET}-{_what}"] = f"robust: held-out blacks, {_name}"
    _recs = sorted((r for r in load_manifest(DATA) if r.set in _by_set and r.id not in _skip
                    and (not r.set.startswith(f"{SIZES_SET}-") or r.parent_id in _held_ids)),   # copies of held-out only
                   key=lambda r: (r.set, r.id))
    for _r in mo.status.progress_bar(_recs, title="folding the stored test sets"):
        _name = f"{_r.id} · {_r.extra.get('original_name') or _r.set}"
        _pr = _r.prompt or (Image.open(resolve(DATA, _r)).getexif().get(272) or _r.set)  # imports: camera model
        _pr = "RGB(" + _pr.split("RGB(", 1)[1].split(")")[0] + ")" if "RGB(" in _pr else _pr[:40]
        _groups[_by_set[_r.set]][1].append((_r.id, _name, _pr, f"{_r.width}x{_r.height}",
                                            file_fold(str(resolve(DATA, _r)), *_args)[0]))

    def _score(f):
        if not np.any(f):
            return 0.0, "flat"
        with np.errstate(invalid="ignore", divide="ignore"):
            cs = {g: fl.corr(f, s) for g, s in verify_sigs.items()}
        cs = {g: (c if np.isfinite(c) else 0.0) for g, c in cs.items()}
        g = max(cs, key=cs.get)
        return float(cs[g]), g

    _syn = [_score(f)[0] for f in synth_folds(*synth_args, None, *_args)]
    threshold = float(np.percentile(_syn, pct_pick.value))
    _ref_sig = verify_sigs[REF]
    _ref_rms = float(np.sqrt(((_ref_sig - _ref_sig.mean()) ** 2).mean()))
    verify_rows = []
    for _grp, (_kind, _items) in _groups.items():
        for _id, _name, _prompt, _size, _f in _items:
            _s, _g = _score(_f)
            _b = fl.slope(_f, _ref_sig)
            verify_rows.append({"group": _grp, "kind": _kind, "prompt": _prompt, "image": _name, "id": _id, "size": _size,
                                "score": round(_s, 3),
                                "best reference": _g, "verdict": "matches Z-Image" if _s > threshold else "no match",
                                "strength": round(_b, 4), "strength (grey levels)": round(_b * _ref_rms, 4)})
    _real = np.array([r["score"] for r in verify_rows if r["kind"] in ("real", "social media")])

    def _auc(pos):
        pos = np.asarray(pos)
        return float(((pos[:, None] > _real[None, :]).sum() + 0.5 * (pos[:, None] == _real[None, :]).sum())
                     / (len(pos) * len(_real)))

    _summary = []
    for _grp, (_kind, _) in _groups.items():
        _sc = [r["score"] for r in verify_rows if r["group"] == _grp]
        if not _sc:
            continue
        _summary.append({"group": _grp, "kind": _kind, "images": len(_sc),
                         "matches Z-Image": sum(s > threshold for s in _sc),
                         "share matched": f"{100 * np.mean([s > threshold for s in _sc]):.0f}%",
                         "median score": round(float(np.median(_sc)), 3), "min": round(min(_sc), 3), "max": round(max(_sc), 3),
                         "AUC vs real": round(_auc(_sc), 3) if _kind not in ("real", "social media") else ""})
    _per_prompt = {}
    for _row in verify_rows:
        _per_prompt.setdefault((_row["group"], _row["prompt"]), []).append(_row["score"])
    _prompt_rows = [{"group": g, "prompt": pr, "images": len(sc), "matches Z-Image": f"{100 * np.mean([x > threshold for x in sc]):.0f}%",
                     "median score": round(float(np.median(sc)), 3), "min": round(min(sc), 3), "max": round(max(sc), 3)}
                    for (g, pr), sc in _per_prompt.items()]
    _style = {"fingerprint test": (CB["vermillion"], "^"), "other generated": (CB["orange"], "^"),
              "real": (CB["blue"], "o"), "social media": (CB["green"], "o"), "unlabelled": (CB["purple"], "D"),
              "AI, other generator": (CB["sky"], "v"), "robustness test": (CB["black"], "x"),
              "FLUX (same VAE)": (CB["vermillion"], "P")}
    _rows = [("synthetic noise (threshold)", _syn, CB["grey"], "s")] + [
        (g, [r["score"] for r in verify_rows if r["group"] == g], *_style[k]) for g, (k, it) in _groups.items() if it]
    verify_fig, _ax = plt.subplots(figsize=(9, 0.55 * len(_rows) + 1.5))
    _jit = np.random.default_rng(1)
    for _i, (_label, _sc, _c, _m) in enumerate(_rows):
        _ax.scatter(_sc, _i + _jit.uniform(-0.22, 0.22, len(_sc)), s=12, color=_c, marker=_m, alpha=0.7)
    _ax.axvline(threshold, color=CB["black"], linestyle="--", linewidth=1, label=f"threshold {threshold:.3f}")
    _ax.set_yticks(range(len(_rows)), [r[0] for r in _rows])
    _ax.invert_yaxis()
    _ax.set_xlabel("score: best aligned correlation with a Z-Image black signature")
    _ax.legend(fontsize=8, loc="lower right")
    verify_fig.tight_layout()
    mo.vstack([mo.md(f"**threshold** {threshold:.3f} ({pct_pick.value}th percentile of {len(_syn)} synthetic scores) · "
                     f"**real** = every real and social-media image ({len(_real)}) · ▲ Z-Image / VAE, ● real / social media, ◆ unlabelled, ▼ AI from other generators"),
               mo.hstack([verify_fig, mo.ui.table(_summary, selection=None, label="per group")], justify="start"),
               mo.ui.table(_prompt_rows, selection=None, page_size=15, label="per prompt (colour, grey level, scene; size for the black groups)"),
               mo.ui.table(verify_rows, selection=None, page_size=20, label="per image (sortable)")])
    return verify_rows, verify_sigs


@app.cell(hide_code=True)
def section_regions():
    mo.md(r"""
    ## 8 · Where in an image does the match come from?

    Each image is scored as in section 7 (best aligned correlation with the held-out-free black signatures), but
    from a fold of **only some of its pixels**: the `share` of pixels that are **darkest** / **brightest** (grey
    level, blurred by the mask σ), **smoothest** / **most detailed** (local RMS of the grey high-pass, same blur),
    and a **random** set of the same size. A fold from fewer pixels is noisier wherever they come from, so compare
    each region with **random**, not with **whole**. Needs section 7 to have run.
    """)
    return


@app.cell(hide_code=True)
def region_controls():
    _sets = sorted({r.set for r in load_manifest(DATA)})
    region_sets = mo.ui.multiselect(_sets, value=[s for s in ("showcase", "adhoc") if s in _sets], label="sets")
    region_frac = mo.ui.slider(0.1, 0.5, value=0.3, step=0.05, label="share of pixels per region", show_value=True)
    region_sigma = mo.ui.slider(1.0, 16.0, value=4.0, step=1.0, label="mask σ (px)", show_value=True)
    region_max = mo.ui.slider(4, 200, value=60, step=4, label="max images per set", show_value=True)
    region_run = mo.ui.run_button(label="score regions")
    mo.hstack([region_sets, region_frac, region_sigma, region_max, region_run], justify="start")
    return region_frac, region_max, region_run, region_sets, region_sigma


@app.cell(hide_code=True)
def region_view(
    mode_pick,
    period_pick,
    region_frac,
    region_max,
    region_run,
    region_sets,
    region_sigma,
    sigma_pick,
    verify_sigs,
):
    mo.stop(not (region_run.value or RUN_ALL), mo.md("Press **score regions** (after section 7 has run)."))
    _p = period_pick.value
    _skip = set()
    for _fp in [*FLAGGED_FILES, *DATA.glob("*-flagged/flagged.md")]:
        if _fp.exists():
            _skip |= read_ids(_fp)
    _recs = []
    for _set in region_sets.value:
        _recs += sorted((r for r in load_manifest(DATA) if r.set == _set and r.id not in _skip),
                        key=lambda r: r.id)[:region_max.value]

    def _best(f):
        if not np.any(f):
            return 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            cs = [fl.corr(f, s) for s in verify_sigs.values()]
        return float(max(c if np.isfinite(c) else 0.0 for c in cs))

    _regions = ("whole", "dark", "bright", "smooth", "detailed", "random")
    region_rows, _preview = [], None
    for _r in mo.status.progress_bar(_recs, title="scoring regions"):
        _px = np.asarray(Image.open(resolve(DATA, _r)).convert("RGB"))
        _px = _px[:(_px.shape[0] // _p) * _p, :(_px.shape[1] // _p) * _p]
        _res = fl.residuals(_px[None], mode_pick.value, None, sigma_pick.value)[0]
        _masks = fl.region_masks(_px, region_frac.value, region_sigma.value)
        _rand = np.zeros(_px.shape[0] * _px.shape[1], bool)
        _rand[np.random.default_rng(int(_r.id[:8], 16)).permutation(_rand.size)[:int(round(region_frac.value * _rand.size))]] = True
        _masks["random"] = _rand.reshape(_px.shape[:2])
        _row = {"set": _r.set, "image": _r.id, "prompt": (_r.prompt or _r.extra.get("original_name") or "")[:48],
                "size": f"{_r.width}x{_r.height}", "whole": round(_best(fl.fold(_res, _p)), 3)}
        for _k in _regions[1:]:
            _row[_k] = round(_best(fl.masked_fold(_res, _masks[_k], _p)), 3)
        region_rows.append(_row)
        if _preview is None:
            _preview = (_px, _masks)
    mo.stop(not region_rows, mo.md("No images in the chosen sets."))
    _by_prompt = {}
    for _row in region_rows:
        _by_prompt.setdefault((_row["set"], _row["prompt"]), []).append(_row)
    _summary = [{"set": k[0], "prompt": k[1], "images": len(v),
                 **{reg: round(float(np.median([r[reg] for r in v])), 3) for reg in _regions}}
                for k, v in _by_prompt.items()]
    _summary.append({"set": "all", "prompt": "", "images": len(region_rows),
                     **{reg: round(float(np.median([r[reg] for r in region_rows])), 3) for reg in _regions}})
    region_fig, _ax = plt.subplots(figsize=(8, 4))
    _cols = [CB["black"], CB["blue"], CB["orange"], CB["green"], CB["vermillion"], CB["grey"]]
    _bp = _ax.boxplot([[r[reg] for r in region_rows] for reg in _regions], showfliers=False, patch_artist=True,
                      medianprops={"color": CB["black"]})
    for _patch, _c in zip(_bp["boxes"], _cols):
        _patch.set(facecolor=_c, alpha=0.6)
    _ax.set_xticks(range(1, len(_regions) + 1), _regions)
    _ax.axhline(0, color=CB["grey"], lw=0.5)
    _ax.set_ylabel("score (best aligned correlation)")
    _ax.set_title(f"{len(region_rows)} images · {region_frac.value:.0%} of pixels per region")
    region_fig.tight_layout()
    _px, _masks = _preview
    _thumb = lambda a: fl.png(np.asarray(Image.fromarray(a).resize((200, round(200 * a.shape[0] / a.shape[1])))))  # noqa: E731
    _shown = [mo.vstack([mo.md("**image**"), mo.image(_thumb(_px))])] + [
        mo.vstack([mo.md(f"**{k}**"), mo.image(_thumb(np.where(m[..., None], _px, (_px * 0.15).astype(np.uint8))))])
        for k, m in _masks.items()]
    mo.vstack([mo.hstack([region_fig, mo.ui.table(_summary, selection=None, page_size=15, label="median per prompt")],
                         justify="start"),
               mo.md("Masks on the first image (chosen pixels kept, the rest dimmed):"),
               mo.hstack(_shown, justify="start", wrap=True),
               mo.ui.table(region_rows, selection=None, page_size=20, label="per image (sortable)")])
    return


@app.cell(hide_code=True)
def section_strength():
    mo.md(r"""
    ## 9 · How strong is the pattern?

    The correlation score says how much an image's fold **looks like** the signature, so the same pattern scores
    lower when more noise sits on top of it. **Strength** measures how much of the pattern is **there**: the
    least-squares slope of each image's fold on one fixed reference, the held-out-free **Z 1024** black signature
    from section 7 (means removed). 1 = as strong as that average clean 1024² black, 0 = none of it.
    **Strength (grey levels)** = slope × the reference's RMS, i.e. the pattern's own size in grey levels. Needs
    section 7 to have run; uses its images and settings.

    **Strength vs real**: each generated group against the real images (every real and social-media group), by **AUC** (how often
    one of its images outscores a real one; 1 = always, 0.5 = chance) for **strength** and, beside it, for the
    **correlation score** of section 7. The **strength threshold** is set like section 7's: the same percentile
    of the synthetic-noise control's strengths, so no test image sets it.
    """)
    return


@app.cell(hide_code=True)
def strength_view(
    mode_pick,
    pct_pick,
    period_pick,
    sigma_pick,
    synth_args,
    synth_folds,
    verify_rows,
    verify_sigs,
):
    _groups = list(dict.fromkeys(r["group"] for r in verify_rows))
    _syn_b = [fl.slope(f, verify_sigs[REF])
              for f in synth_folds(*synth_args, None, mode_pick.value, sigma_pick.value, period_pick.value)]
    strength_threshold = float(np.percentile(_syn_b, pct_pick.value))
    _is_real = lambda r: r["kind"] in ("real", "social media")  # noqa: E731

    def _auc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                     / (len(pos) * len(neg)))

    _real_b = [r["strength"] for r in verify_rows if _is_real(r)]
    _real_c = [r["score"] for r in verify_rows if _is_real(r)]
    strength_auc = []
    for _g in _groups:
        _rs = [r for r in verify_rows if r["group"] == _g]
        _b = [r["strength"] for r in _rs]
        _row = {"group": _g, "kind": _rs[0]["kind"], "images": len(_rs),
                "over strength threshold": f"{100 * np.mean([x > strength_threshold for x in _b]):.0f}%"}
        if not _is_real(_rs[0]):
            _row["AUC strength"] = round(_auc(_b, _real_b), 3)
            _row["AUC correlation"] = round(_auc([r["score"] for r in _rs], _real_c), 3)
        strength_auc.append(_row)
    _per = {}
    for _row in verify_rows:
        _per.setdefault((_row["group"], _row["prompt"]), []).append(_row)

    def _stats(rows):
        b = np.array([r["strength"] for r in rows])
        g = np.array([r["strength (grey levels)"] for r in rows])
        return {"images": len(rows), "median strength": round(float(np.median(b)), 3),
                "p10": round(float(np.percentile(b, 10)), 3), "p90": round(float(np.percentile(b, 90)), 3),
                "median grey levels": round(float(np.median(g)), 4)}

    _by_group = [{"group": g, **_stats([r for r in verify_rows if r["group"] == g])} for g in _groups]
    _by_prompt = [{"group": g, "prompt": pr, **_stats(v)} for (g, pr), v in _per.items()]
    _kind = {r["group"]: r["kind"] for r in verify_rows}
    _col = {"fingerprint test": CB["vermillion"], "other generated": CB["orange"], "real": CB["blue"],
            "social media": CB["green"], "unlabelled": CB["purple"], "AI, other generator": CB["sky"],
            "robustness test": CB["black"], "FLUX (same VAE)": CB["vermillion"]}
    strength_fig, _ax = plt.subplots(figsize=(9, 0.55 * len(_groups) + 1.5))
    _jit = np.random.default_rng(2)
    for _i, _g in enumerate(_groups):
        _b = [r["strength"] for r in verify_rows if r["group"] == _g]
        _ax.scatter(_b, _i + _jit.uniform(-0.22, 0.22, len(_b)), s=12, color=_col[_kind[_g]], alpha=0.7)
    _ax.axvline(0, color=CB["grey"], lw=0.8)
    _ax.axvline(1, color=CB["black"], lw=0.8, linestyle="--", label="1 = the Z 1024 black signature")
    _ax.axvline(strength_threshold, color=CB["vermillion"], lw=0.8, linestyle=":",
                label=f"strength threshold {strength_threshold:.3f}")
    _ax.set_yticks(range(len(_groups)), _groups)
    _ax.invert_yaxis()
    _ax.set_xlabel(f"strength: slope of the image's fold on the {REF} signature")
    _ax.legend(fontsize=8, loc="lower right")
    strength_fig.tight_layout()
    mo.vstack([mo.md(f"**strength threshold** {strength_threshold:.4f} ({pct_pick.value}th percentile of "
                     f"{len(_syn_b)} synthetic strengths) · **real** = every real and social-media image ({len(_real_b)})"),
               mo.hstack([strength_fig, mo.ui.table(_by_group, selection=None, page_size=20, label="per group")],
                         justify="start"),
               mo.ui.table(strength_auc, selection=None, page_size=20, label="strength vs real (AUC beside the correlation's)"),
               mo.ui.table(_by_prompt, selection=None, page_size=20, label="per prompt (colour, grey level, scene; size for the black groups)")])
    return strength_auc, strength_threshold


if __name__ == "__main__":
    app.run()
