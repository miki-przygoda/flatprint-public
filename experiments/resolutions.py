import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import functools
    import os
    import sys
    from fractions import Fraction
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    import cleanup_lib as cl
    import fingerprint_lib as fl
    from flatprint.dataset import load_matrix, read_ids
    from flatprint.store import load_manifest

    DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
    SET = "flat-black-sizes"
    FLAGGED = DATA / f"{SET}-flagged" / "flagged.md"
    RGB_FLAGGED = DATA / "flat-rgb-flagged" / "flagged.md"
    OLD = "old 1024 (flat-rgb)"
    TARGET = (0, 0, 0)
    EDGE_MARGIN = 1.5            # same default as cleanup.py: over 1.5 x the worst reference score
    try:                         # `-- --all` (e.g. marimo export html NB -- --all): press every run button
        RUN_ALL = "all" in mo.cli_args()
    except Exception:            # imported as a module, outside marimo: read the command line itself
        RUN_ALL = "--all" in sys.argv
    # Okabe-Ito palette: stays distinct under the common colour-vision deficiencies. Heatmaps use magma
    # (sequential) and RdBu (diverging), which are colour-blind safe too.
    CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
          "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#999999", "black": "#000000"}
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[CB[k] for k in ("blue", "orange", "green", "vermillion",
                                                                         "purple", "sky", "grey", "black")])


@app.cell
def intro():
    mo.md(r"""
    # Black at four resolutions

    Five **groups** of black RGB(0, 0, 0) images: the `flat-black-sizes` set at 512², 768², 1024² and 1280²
    (every seed drawn at all four sizes), and the 32 + flagged blacks from the earlier `flat-rgb` run
    (**old 1024**).

    - **flagged**: in `data/flat-black-sizes-flagged/flagged.md` or `data/flat-rgb-flagged/flagged.md`; left out
      of fingerprints unless the switch is on.
    - **edge case**: over a `cleanup.py` threshold (reference: clean old blacks, margin 1.5) but kept, i.e. not
      flagged. They stay in the fingerprints and are also scored as test images in section 8.

    Spectrum positions are in **cycles per pixel** (`fx·16/W`, sixteenths) and **cycles per image** (`fx`).
    A pattern with a fixed period in pixels keeps its cycles-per-pixel position at every size.
    """)
    return


@app.cell
def load_data():
    _sizes = sorted({(r.width, r.height) for r in load_manifest(DATA) if r.set == SET})
    groups = {}
    with mo.status.spinner(f"loading {SET} and the old blacks (the first time builds the cache)"):
        for _s in _sizes:
            _m = load_matrix(DATA, SET, size=_s)
            groups[f"{_s[0]}x{_s[1]}"] = SimpleNamespace(size=_s, pixels=_m.pixels, rows=list(range(len(_m.records))),
                                                         records=_m.records)
        try:
            _ref = load_matrix(DATA, "flat-rgb")
            _rows = [i for i, n in enumerate(_ref.names) if n == "black"]
            groups[OLD] = SimpleNamespace(size=(_ref.pixels.shape[2], _ref.pixels.shape[1]), pixels=_ref.pixels,
                                          rows=_rows, records=[_ref.records[i] for i in _rows])
        except (ValueError, IndexError):
            pass
    new_groups = [g for g in groups if g != OLD]
    row_of_seed = {g: {r.seed: k for k, r in enumerate(groups[g].records)} for g in new_groups}
    paired_seeds = sorted(set.intersection(*(set(d) for d in row_of_seed.values())))
    return groups, new_groups, paired_seeds, row_of_seed


@app.cell
def reload_button():
    reload_pick = mo.ui.button(value=0, on_click=lambda v: v + 1, label="reload flagged lists")
    reload_pick
    return (reload_pick,)


@app.cell
def flagged_lists(groups, reload_pick):
    reload_pick.value
    _flagged = set()
    for _p in (FLAGGED, RGB_FLAGGED):
        if _p.exists():
            _flagged |= read_ids(_p)
    is_flagged = {g: np.array([r.id in _flagged for r in G.records]) for g, G in groups.items()}
    return (is_flagged,)


@app.cell
def edge_cases(groups, is_flagged):
    _old = groups[OLD]
    _ref = np.stack([cl.thumbnail(_old.pixels[_old.rows[k]]) for k in range(len(_old.rows)) if not is_flagged[OLD][k]])
    _cal = cl.calibrate(_ref)
    _limit = {k: EDGE_MARGIN * float(_cal[:, j].max()) for j, k in enumerate(cl.SCORES)}
    _mean = _ref.mean(axis=0)
    is_edge, edge_reason = {}, {}
    for _g, _G in mo.status.progress_bar(list(groups.items()), title="finding edge cases"):
        _edge, _why = np.zeros(len(_G.rows), bool), {}
        if _g != OLD:
            for _k in range(len(_G.rows)):
                if is_flagged[_g][_k]:
                    continue
                _sc = cl.scores(cl.thumbnail(_G.pixels[_G.rows[_k]]), _mean)
                _over = [s for s in cl.SCORES if _sc[s] > _limit[s]]
                if _over:
                    _edge[_k], _why[_k] = True, ", ".join(_over)
        is_edge[_g], edge_reason[_g] = _edge, _why
    mo.ui.table([{"group": g, "images": len(G.rows), "flagged": int(is_flagged[g].sum()), "edge cases": int(is_edge[g].sum()),
                  "clean (in fingerprints)": int((~is_flagged[g]).sum())} for g, G in groups.items()],
                selection=None, label="groups")
    return edge_reason, is_edge


@app.cell
def controls(groups):
    group_pick = mo.ui.dropdown(list(groups), value="1024x1024" if "1024x1024" in groups else next(iter(groups)),
                                label="focus group")
    mode_pick = mo.ui.radio(["mean", "target", "highpass"], value="highpass", inline=True, label="residual")
    sigma_pick = mo.ui.slider(1.0, 10.0, value=3.0, step=0.5, label="highpass blur σ (px)", show_value=True)
    flagged_pick = mo.ui.switch(value=False, label="include flagged images in fingerprints")
    period_pick = mo.ui.dropdown({"8": 8, "16": 16, "32": 32}, value="16", label="fold period (px)")
    k_pick = mo.ui.slider(1, 10, value=4, label="stretch ±k std", show_value=True)
    hann_pick = mo.ui.switch(value=True, label="Hann window")
    guides_pick = mo.ui.switch(value=True, label="1/8 + 1/16 guides")
    clip_pick = mo.ui.slider(90.0, 100.0, value=99.9, step=0.1, label="spectrum clip %", show_value=True)
    mo.vstack([
        mo.hstack([group_pick, mode_pick, sigma_pick, flagged_pick, period_pick], justify="start"),
        mo.hstack([k_pick, hann_pick, guides_pick, clip_pick], justify="start"),
        mo.md("**residual**: `mean` = minus the image's mean colour · `target` = minus RGB(0, 0, 0) · "
              "`highpass` = minus a blurred copy (also removes large-scale shading; σ sets the blur)"),
    ])
    return (
        clip_pick,
        flagged_pick,
        group_pick,
        guides_pick,
        hann_pick,
        k_pick,
        mode_pick,
        period_pick,
        sigma_pick,
    )


@app.cell
def spectrum_fn(clip_pick, guides_pick, hann_pick):
    def spectrum_view(img):
        spec = fl.log_spectrum(img, hann_pick.value)
        shown = fl.show_spectrum(spec, clip_pick.value)
        return spec, (fl.with_guides(shown) if guides_pick.value else shown)

    return (spectrum_view,)


@app.cell
def pass_cache(groups):
    @functools.lru_cache(maxsize=4)
    def residual_pass(mode, sigma, period, flagged_key):
        """One pass over every image: per group the residual sum over all images and over the clean ones
        (full size), and every image's fold. flagged_key = the flagged ids, so a reload recomputes."""
        out = {}
        for g, G in groups.items():
            n = len(G.rows)
            folds = np.empty((n, period, period, 3), np.float32)
            total = np.zeros(G.pixels.shape[1:], np.float64)
            clean = np.zeros_like(total)
            for a in range(0, n, 16):
                res = fl.residuals(G.pixels[G.rows[a:a + 16]], mode, TARGET, sigma)
                folds[a:a + 16] = fl.fold(res, period)
                for j, r in enumerate(res):
                    total += r
                    if G.records[a + j].id not in flagged_key:
                        clean += r
            n_clean = sum(r.id not in flagged_key for r in G.records)
            out[g] = {"folds": folds, "total": total, "n": n, "clean": clean, "n_clean": n_clean}
        return out

    return (residual_pass,)


@app.cell
def compute_pass(
    flagged_pick,
    groups,
    is_flagged,
    mode_pick,
    period_pick,
    residual_pass,
    sigma_pick,
):
    _key = frozenset(r.id for g, G in groups.items() for r, f in zip(G.records, is_flagged[g]) if f)
    with mo.status.spinner(f"{mode_pick.value} residuals for every image (fingerprints and {period_pick.value}-px folds)"):
        passes = residual_pass(mode_pick.value, sigma_pick.value, period_pick.value, _key)
    fp_sums = {g: ((p["total"], p["n"]) if flagged_pick.value else (p["clean"], p["n_clean"])) for g, p in passes.items()}
    fps = {g: s / n for g, (s, n) in fp_sums.items()}
    folds = {g: p["folds"] for g, p in passes.items()}
    return folds, fp_sums, fps


@app.cell
def section_browse():
    mo.md(r"""
    ## 1 · Browse (one seed at every size)

    Top row: the images as generated. Bottom row: stretched (each image's own mean removed, ±k std).
    """)
    return


@app.cell
def browse_controls(paired_seeds):
    seed_pick = mo.ui.slider(0, len(paired_seeds) - 1, value=0, label="seed", show_value=True)
    seed_pick
    return (seed_pick,)


@app.cell
def browse(
    groups,
    is_edge,
    is_flagged,
    k_pick,
    new_groups,
    paired_seeds,
    row_of_seed,
    seed_pick,
):
    _seed = paired_seeds[seed_pick.value]
    _cols = []
    for _g in new_groups:
        _k = row_of_seed[_g][_seed]
        _G = groups[_g]
        _rec, _px = _G.records[_k], _G.pixels[_G.rows[_k]]
        _img, _lv = fl.stretched(_px, k_pick.value)
        _tag = " · flagged" if is_flagged[_g][_k] else (" · edge case" if is_edge[_g][_k] else "")
        _mean = ", ".join(f"{v:.1f}" for v in _px.mean(axis=(0, 1), dtype=np.float64))
        _cols.append(mo.vstack([mo.md(f"**{_g}**{_tag}  \n`{_rec.id}` · mean {_mean} · ±{_lv:.2f}"),
                                mo.image(fl.png(_px), width=300), mo.image(fl.png(_img), width=300)]))
    mo.vstack([mo.md(f"seed `{_seed}`"), mo.hstack(_cols, justify="start")])
    return


@app.cell
def section_outliers():
    mo.md(r"""
    ## 2 · Outlier finder (every image)

    Uses mean-colour residuals regardless of the setting above. Bands scale with the image so sizes compare.

    - **mean RGB**, **off target**: the image's mean colour and its distance from RGB(0, 0, 0).
    - **noise**: standard deviation of the residual.
    - **top band**: brightness of the top 1/16 of rows minus the whole image (grey levels).
    - **edge**: brightness of the outer 1/64 border minus the inside (16 px at 1024).
    - **corr**: correlation with its group's fingerprint (leave-one-out, mean residuals, flagged excluded).
    """)
    return


@app.cell
def outliers(groups, is_edge, is_flagged):
    outlier_rows = []
    for _g, _G in mo.status.progress_bar(list(groups.items()), title="measuring every image"):
        _clean = ~is_flagged[_g]
        _total, _n = fl.fingerprint_sum(_G.pixels, [_G.rows[k] for k in np.flatnonzero(_clean)], "mean")
        _band, _b = _G.size[1] // 16, _G.size[0] // 64
        for _k, _rec in enumerate(_G.records):
            _px = _G.pixels[_G.rows[_k]].astype(np.float64)
            _mean = _px.mean(axis=(0, 1))
            _r = _px - _mean
            _grey = _px.mean(axis=2)
            _inner = _grey[_b:-_b, _b:-_b]
            _border = (_grey.sum() - _inner.sum()) / (_grey.size - _inner.size)
            _ref = fl.leave_one_out(_total, _n, _r) if _clean[_k] else _total / _n
            outlier_rows.append({
                "group": _g, "seed": _rec.seed, "id": _rec.id, "flagged": "yes" if is_flagged[_g][_k] else "",
                "edge case": "yes" if is_edge[_g][_k] else "",
                "mean RGB": ", ".join(f"{v:.1f}" for v in _mean),
                "off target": round(float(np.linalg.norm(_mean)), 2),
                "noise": round(float(_r.std()), 3),
                "top band": round(float(_grey[:_band].mean() - _grey.mean()), 3),
                "edge": round(float(_border - _inner.mean()), 3),
                "corr": round(fl.corr(_r, _ref), 4)})
    mo.ui.table(outlier_rows, selection=None, page_size=30, label=f"{len(outlier_rows)} images (sort any column)")
    return


@app.cell
def section_fingerprints():
    mo.md(r"""
    ## 3 · Fingerprint per group

    The average residual per group (stretched) and its spectrum. The guides sit at the same cycles-per-pixel
    positions at every size. Old and new 1024² have the same size, so they can also be compared pixel by pixel.
    """)
    return


@app.cell
def fingerprint_view(fp_sums, fps, groups, k_pick, spectrum_view):
    _cols = []
    for _g in groups:
        _img, _lv = fl.stretched(fps[_g], k_pick.value)
        _, _spec_img = spectrum_view(fps[_g])
        _cols.append(mo.vstack([mo.md(f"**{_g}** · n={fp_sums[_g][1]} · ±{_lv:.2f} levels"),
                                mo.image(fl.png(_img), width=260), mo.image(fl.png(_spec_img), width=260)]))
    _line = (f"new 1024² vs old 1024² fingerprint: corr **{fl.corr(fps['1024x1024'], fps[OLD]):.4f}**"
             if "1024x1024" in fps and OLD in fps else "")
    mo.vstack([mo.hstack(_cols, justify="start"), mo.md(_line)])
    return


@app.cell
def section_peaks():
    mo.md(r"""
    ## 4 · Peaks across groups

    The top-N peaks of each group's fingerprint spectrum, matched across groups two ways (exact FFT bins):
    **by cycles per pixel** (same period in pixels) and **by cycles per image** (same repeats across the frame).
    """)
    return


@app.cell
def peak_controls():
    npeaks_pick = mo.ui.slider(5, 50, value=20, label="top N peaks per group", show_value=True)
    min_groups_pick = mo.ui.slider(1, 5, value=2, label="appears in at least N groups", show_value=True)
    mo.hstack([npeaks_pick, min_groups_pick], justify="start")
    return min_groups_pick, npeaks_pick


@app.cell
def peaks(fps, groups, min_groups_pick, npeaks_pick, spectrum_view):
    peaks_by_group = {g: fl.find_peaks(spectrum_view(fps[g])[0], npeaks_pick.value, 7, False) for g in groups}
    _by_cpp, _by_cpi = {}, {}
    for _g, _rows in peaks_by_group.items():
        _w, _h = groups[_g].size
        for _p in _rows:
            _by_cpp.setdefault((Fraction(_p["fx"] * 16, _w), Fraction(_p["fy"] * 16, _h)), []).append(_g)
            _by_cpi.setdefault((_p["fx"], _p["fy"]), []).append(_g)

    def _period(f16):
        return round(16 / abs(float(f16)), 2) if f16 else "∞"

    peaks_cpp = sorted(({"fx·16/W": round(float(fx), 3), "fy·16/H": round(float(fy), 3),
                         "period x (px)": _period(fx), "period y (px)": _period(fy),
                         "on 1/16 lattice": "yes" if fx.denominator == 1 and fy.denominator == 1 else "",
                         "groups": len(v), "which": ", ".join(v)}
                        for (fx, fy), v in _by_cpp.items() if len(v) >= min_groups_pick.value),
                       key=lambda r: -r["groups"])
    peaks_cpi = sorted(({"fx": fx, "fy": fy, "groups": len(v), "which": ", ".join(v)}
                        for (fx, fy), v in _by_cpi.items() if len(v) >= min_groups_pick.value),
                       key=lambda r: -r["groups"])
    mo.ui.tabs({
        f"by cycles per pixel ({len(peaks_cpp)})": mo.ui.table(peaks_cpp, selection=None, page_size=30),
        f"by cycles per image ({len(peaks_cpi)})": mo.ui.table(peaks_cpi, selection=None, page_size=30),
        **{f"{g} top {npeaks_pick.value}": mo.ui.table(peaks_by_group[g], selection=None, page_size=50) for g in groups},
    })
    return


@app.cell
def section_tracker():
    mo.md(r"""
    ## 5 · Lattice tracker across groups

    Local contrast at every point of the 1/16 cycles-per-pixel lattice, per group. Columns are the top 30 by
    mean contrast across groups. Below: the same positions in every image of the focus group.
    """)
    return


@app.cell
def tracker(fps, group_pick, groups, mode_pick, sigma_pick, spectrum_view):
    _names = list(groups)
    _pts = {g: fl.lattice(*groups[g].size, div=16) for g in _names}
    _w0, _h0 = groups[_names[0]].size
    _keys = [(fx * 16 // _w0, fy * 16 // _h0) for fx, fy in _pts[_names[0]]]
    _strength = np.array([fl.lattice_strength(spectrum_view(fps[g])[0], _pts[g]) for g in _names])
    _order = np.argsort(_strength.mean(axis=0))[::-1]
    tracker_rows = [{"fx·16/W": _keys[j][0], "fy·16/H": _keys[j][1], "mean": round(float(_strength[:, j].mean()), 3),
                     **{g: round(float(_strength[i, j]), 2) for i, g in enumerate(_names)}} for j in _order]
    _top = _order[:30]
    _labels = [f"({_keys[j][0]},{_keys[j][1]})" for j in _top]
    _G = groups[group_pick.value]
    _focus_pts = [_pts[group_pick.value][j] for j in _top]
    _per_image = np.array([fl.lattice_strength(spectrum_view(
        fl.residuals(_G.pixels[[_G.rows[k]]], mode_pick.value, TARGET, sigma_pick.value)[0])[0], _focus_pts)
        for k in mo.status.progress_bar(range(len(_G.rows)), title=f"lattice in every {group_pick.value} image")])
    tracker_fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 3])
    _i1 = _a1.imshow(_strength[:, _top], aspect="auto", cmap="magma")
    _a1.set_yticks(range(len(_names)), _names)
    _a1.set_xticks(range(len(_top)), _labels, rotation=90, fontsize=7)
    _a1.set_title("contrast at the top-30 lattice positions per group fingerprint (positions in 1/16 cycles per pixel)")
    tracker_fig.colorbar(_i1, ax=_a1)
    _i2 = _a2.imshow(_per_image, aspect="auto", cmap="magma", interpolation="nearest")
    _a2.set_ylabel(f"{group_pick.value} images (rows, by seed)")
    _a2.set_xticks(range(len(_top)), _labels, rotation=90, fontsize=7)
    _a2.set_title(f"same positions in every {group_pick.value} image")
    tracker_fig.colorbar(_i2, ax=_a2)
    tracker_fig.tight_layout()
    mo.vstack([tracker_fig, mo.ui.table(tracker_rows, selection=None, page_size=20,
                                        label=f"{len(tracker_rows)} lattice positions, sorted by mean contrast")])
    return


@app.cell
def section_fold():
    mo.md(r"""
    ## 6 · Fold signatures

    Each image's residual folded into one **period × period tile** (the average of all its tiles), so only what
    repeats every `period` pixels and sits at the same place in each tile survives. A group's **signature** is
    the mean fold of its clean images. Every size is a multiple of 32, so tiles compare across groups.

    - **Signatures** side by side (stretched), and the correlation between groups: as they are (**aligned**),
      the **best over every cyclic shift** of the tile (and which shift), and the **median over shifts** (a
      control: what unrelated alignments give).
    - **Per image** (focus group): each fold against its own group's signature without it (**own, LOO**),
      against the pooled signature of the **other groups**, and against its own signature **shifted** by a
      random non-zero offset (control).
    """)
    return


@app.cell
def signatures(folds, groups, is_flagged, k_pick):
    signature = {g: folds[g][~is_flagged[g]].mean(axis=0) for g in groups}
    _names = list(groups)
    _p = next(iter(signature.values())).shape[0]
    _shifts = [(dy, dx) for dy in range(_p) for dx in range(_p)]
    fold_pairs = []
    _aligned = np.eye(len(_names))
    for _i, _a in enumerate(_names):
        for _j, _b in enumerate(_names):
            if _j <= _i:
                continue
            _c = np.array([fl.corr(signature[_a], np.roll(signature[_b], s, axis=(0, 1))) for s in _shifts])
            _aligned[_i, _j] = _aligned[_j, _i] = _c[0]
            _best = int(np.argmax(_c))
            fold_pairs.append({"a": _a, "b": _b, "aligned": round(float(_c[0]), 4), "best over shifts": round(float(_c[_best]), 4),
                               "best shift (dy, dx)": str(_shifts[_best]), "median over shifts": round(float(np.median(_c)), 4)})
    _scale = max(1, 192 // _p)
    _tiles = [mo.vstack([mo.md(f"**{g}** · ±{fl.stretched(signature[g], k_pick.value)[1]:.3f}"),
                         mo.image(fl.png(fl.zoom(fl.stretched(signature[g], k_pick.value)[0], 0, 0, _p, _scale)),
                                  width=_p * _scale)]) for g in _names]
    fold_fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.imshow(_aligned, cmap="RdBu_r", vmin=-1, vmax=1)
    _ax.set_xticks(range(len(_names)), _names, rotation=45, ha="right")
    _ax.set_yticks(range(len(_names)), _names)
    for (_i, _j), _v in np.ndenumerate(_aligned):
        _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=8)
    fold_fig.colorbar(_im, ax=_ax, shrink=0.8)
    _ax.set_title(f"{_p}-px fold signature correlation (aligned)")
    fold_fig.tight_layout()
    mo.vstack([mo.hstack(_tiles, justify="start"),
               mo.hstack([fold_fig, mo.ui.table(fold_pairs, selection=None, page_size=12)], justify="start")])
    return (signature,)


@app.cell
def fold_per_image(folds, group_pick, groups, is_edge, is_flagged):
    _g = group_pick.value
    _F, _flag = folds[_g], is_flagged[_g]
    _clean = np.flatnonzero(~_flag)
    _sum, _n = _F[_clean].astype(np.float64).sum(axis=0), len(_clean)
    _others = np.concatenate([folds[h][~is_flagged[h]] for h in groups if h != _g]).mean(axis=0)
    _p = _F.shape[1]
    _rng = np.random.default_rng(0)
    fold_scores = []
    for _k, _rec in enumerate(groups[_g].records):
        _own = (_sum - _F[_k]) / (_n - 1) if not _flag[_k] else _sum / _n
        _s = tuple(int(v) for v in _rng.integers(1, _p, 2))
        fold_scores.append({"seed": _rec.seed, "id": _rec.id, "flagged": "yes" if _flag[_k] else "",
                            "edge case": "yes" if is_edge[_g][_k] else "",
                            "own (LOO)": round(fl.corr(_F[_k], _own), 4), "other groups": round(fl.corr(_F[_k], _others), 4),
                            "shifted control": round(fl.corr(_F[_k], np.roll(_own, _s, axis=(0, 1))), 4)})
    _cl = [r for r in fold_scores if not r["flagged"]]
    fold_hist, _ax = plt.subplots(figsize=(7, 3.5))
    _bins = np.linspace(-1, 1, 81)
    for _col, _c in (("own (LOO)", CB["blue"]), ("other groups", CB["orange"]), ("shifted control", CB["grey"])):
        _ax.hist([r[_col] for r in _cl], bins=_bins, alpha=0.5, color=_c, label=f"{_col} (median {np.median([r[_col] for r in _cl]):.3f})")
    _ax.set_xlabel("correlation of an image's fold with a signature")
    _ax.set_ylabel("images")
    _ax.set_title(f"{_g}: clean images ({len(_cl)})")
    _ax.legend(fontsize=8)
    fold_hist.tight_layout()
    mo.hstack([fold_hist, mo.ui.table(fold_scores, selection=None, page_size=20, label=f"{_g} folds (sortable)")],
              justify="start")
    return


@app.cell
def section_seeds():
    mo.md(r"""
    ## 7 · Same seed across sizes

    Every new seed was drawn at all four sizes. For each pair of sizes: the correlation between the **same
    seed's** folds, against **different seeds'** folds (each seed paired with a random other seed). Only seeds
    clean at both sizes. **Remove group signature** subtracts each group's signature first, so what is left is
    what is particular to the seed rather than what every image shares.
    """)
    return


@app.cell
def seed_controls():
    remove_sig_pick = mo.ui.switch(value=False, label="remove group signature")
    remove_sig_pick
    return (remove_sig_pick,)


@app.cell
def same_seed(
    folds,
    is_flagged,
    new_groups,
    remove_sig_pick,
    row_of_seed,
    signature,
):
    _rng = np.random.default_rng(0)

    def _fold(g, seed):
        _f = folds[g][row_of_seed[g][seed]]
        return _f - signature[g] if remove_sig_pick.value else _f

    seed_pairs, _box, _box_labels = [], [], []
    _per_seed = {}
    for _i, _a in enumerate(new_groups):
        for _b in new_groups[_i + 1:]:
            _seeds = [s for s in row_of_seed[_a] if s in row_of_seed[_b]
                      and not is_flagged[_a][row_of_seed[_a][s]] and not is_flagged[_b][row_of_seed[_b][s]]]
            if len(_seeds) < 3:
                continue
            _same = [fl.corr(_fold(_a, s), _fold(_b, s)) for s in _seeds]
            _perm = _rng.permutation(len(_seeds))
            _perm = np.where(_perm == np.arange(len(_seeds)), np.roll(_perm, 1), _perm)
            _diff = [fl.corr(_fold(_a, s), _fold(_b, _seeds[j])) for s, j in zip(_seeds, _perm)]
            for s, c in zip(_seeds, _same):
                _per_seed.setdefault(s, []).append(c)
            seed_pairs.append({"sizes": f"{_a} · {_b}", "seeds": len(_seeds),
                               "same seed median": round(float(np.median(_same)), 4), "same seed mean": round(float(np.mean(_same)), 4),
                               "different seeds median": round(float(np.median(_diff)), 4),
                               "different seeds mean": round(float(np.mean(_diff)), 4)})
            _box += [_same, _diff]
            _box_labels += [f"{_a}·{_b}\nsame", "diff"]
    seed_fig, _ax = plt.subplots(figsize=(12, 4))
    _bp = _ax.boxplot(_box, showfliers=False, patch_artist=True, medianprops={"color": CB["black"]})
    for _i, _patch in enumerate(_bp["boxes"]):
        _patch.set(facecolor=CB["blue"] if _i % 2 == 0 else CB["orange"], alpha=0.7)   # same seed, then different seeds
    _ax.set_xticks(range(1, len(_box) + 1), _box_labels, fontsize=7)
    _ax.axhline(0, color=CB["grey"], lw=0.5)
    _ax.set_ylabel("fold correlation")
    _ax.set_title("same seed vs different seeds, per pair of sizes" + (" (group signature removed)" if remove_sig_pick.value else ""))
    seed_fig.tight_layout()
    seed_rows = sorted(({"seed": s, "size pairs": len(v), "mean corr": round(float(np.mean(v)), 4)} for s, v in _per_seed.items()),
                       key=lambda r: -r["mean corr"])
    mo.vstack([mo.ui.table(seed_pairs, selection=None), seed_fig,
               mo.ui.table(seed_rows, selection=None, page_size=15, label="per seed: mean same-seed corr over its size pairs")])
    return


@app.cell
def section_tests():
    mo.md(r"""
    ## 8 · Test images: edge cases and flagged

    Your kept edge cases and the flagged images, scored against the **clean** fingerprint of their group, as
    an early look at what a detector would see. Edge cases are part of the fingerprint, so they are scored
    leave-one-out; flagged images against the fingerprint as it stands.

    - **fold corr** and its **percentile** among the group's other clean (non-edge) images' fold corrs.
    - **full corr**: full-size residual against the fingerprint. **shared peaks**: of the image's top 20 with
      the fingerprint's top 20.
    """)
    return


@app.cell
def test_run_button():
    test_run = mo.ui.run_button(label="score test images")
    test_run
    return (test_run,)


@app.cell
def test_images(
    edge_reason,
    folds,
    fp_sums,
    groups,
    is_edge,
    is_flagged,
    mode_pick,
    sigma_pick,
    signature,
    spectrum_view,
    test_run,
):
    mo.stop(not (test_run.value or RUN_ALL), mo.md("Press **score test images** (about a minute with highpass residuals)."))
    _todo = [(g, k) for g in groups for k in range(len(groups[g].rows)) if is_edge[g][k] or is_flagged[g][k]]
    test_rows = []
    _ref_pts = {}
    _base = {}
    for _g in groups:
        _F, _ok = folds[_g], ~is_flagged[_g]
        _s, _n = _F[_ok].astype(np.float64).sum(axis=0), int(_ok.sum())
        _plain = np.flatnonzero(_ok & ~is_edge[_g])
        _base[_g] = [fl.corr(_F[k], (_s - _F[k]) / (_n - 1)) for k in _plain]
        _ref_pts[_g] = {(p["fx"], p["fy"]) for p in fl.find_peaks(spectrum_view(fp_sums[_g][0] / fp_sums[_g][1])[0], 20, 7, False)}
    for _g, _k in mo.status.progress_bar(_todo, title="scoring test images"):
        _G = groups[_g]
        _F, _ok = folds[_g], ~is_flagged[_g]
        _n = int(_ok.sum())
        _own = (signature[_g] * _n - _F[_k]) / (_n - 1) if _ok[_k] else signature[_g]
        _fc = fl.corr(_F[_k], _own)
        _r = fl.residuals(_G.pixels[[_G.rows[_k]]], mode_pick.value, TARGET, sigma_pick.value)[0]
        _t, _tn = fp_sums[_g]
        _ref = fl.leave_one_out(_t, _tn, _r) if _ok[_k] else _t / _tn
        _pts = {(p["fx"], p["fy"]) for p in fl.find_peaks(spectrum_view(_r)[0], 20, 7, False)}
        _rec = _G.records[_k]
        test_rows.append({"group": _g, "kind": "edge case" if is_edge[_g][_k] else "flagged",
                          "edge reason": edge_reason[_g].get(_k, ""), "seed": _rec.seed, "id": _rec.id,
                          "fold corr": round(_fc, 4), "fold percentile": round(fl.percentile_of(_fc, _base[_g]), 1),
                          "full corr": round(fl.corr(_r, _ref), 4), "shared peaks": len(_pts & _ref_pts[_g])})
    _summary = [{"group": g, "clean non-edge images": len(_base[g]),
                 "their fold corr median": round(float(np.median(_base[g])), 4) if _base[g] else None,
                 "edge cases": sum(r["kind"] == "edge case" and r["group"] == g for r in test_rows),
                 "flagged": sum(r["kind"] == "flagged" and r["group"] == g for r in test_rows)} for g in groups]
    mo.vstack([mo.ui.table(_summary, selection=None, label="baseline per group"),
               mo.ui.table(test_rows, selection=None, page_size=30, label=f"{len(test_rows)} test images (sortable)")])
    return


@app.cell
def section_curve():
    mo.md(r"""
    ## 9 · Sample-size curve per group

    Two disjoint random sets of N clean images, each averaged, then correlated; repeated for each N.
    Dashed: the shifted control. One group at a time, so the first run for a setting takes a while.
    """)
    return


@app.cell
def curve_controls():
    repeats_pick = mo.ui.slider(3, 30, value=10, label="repeats per N", show_value=True)
    curve_run = mo.ui.run_button(label="run curves")
    mo.hstack([repeats_pick, curve_run], justify="start")
    return curve_run, repeats_pick


@app.cell
def curve_cache(groups):
    @functools.lru_cache(maxsize=8)
    def curves(mode, sigma, repeats, flagged_key):
        """Aligned and shifted split-half correlations per group and N (flagged images left out)."""
        out = {}
        for g, G in groups.items():
            idx = [G.rows[k] for k, r in enumerate(G.records) if r.id not in flagged_key]
            res = np.empty((len(idx),) + G.pixels.shape[1:], dtype=np.float32)
            for a in range(0, len(idx), 16):
                res[a:a + 16] = fl.residuals(G.pixels[idx[a:a + 16]], mode, TARGET, sigma)
            ns = [n for n in (1, 2, 4, 8, 16, 32, 64) if 2 * n <= len(res)]
            rng = np.random.default_rng(0)
            aligned = fl.split_half_curve(res, ns, repeats, rng)
            shifted = {}
            for n in ns:
                vals = []
                for _ in range(repeats):
                    o = rng.permutation(len(res))
                    a_, b_ = res[o[:n]].mean(0), res[o[n:2 * n]].mean(0)
                    dy, dx = rng.integers(16, a_.shape[0] - 16), rng.integers(16, a_.shape[1] - 16)
                    vals.append(fl.corr(a_, np.roll(b_, (dy, dx), axis=(0, 1))))
                shifted[n] = vals
            out[g] = (aligned, shifted)
            del res
        return out

    return (curves,)


@app.cell
def curve_view(
    curve_run,
    curves,
    groups,
    is_flagged,
    mode_pick,
    repeats_pick,
    sigma_pick,
):
    mo.stop(not (curve_run.value or RUN_ALL), mo.md("Press **run curves** (about a minute per group with highpass residuals)."))
    _key = frozenset(r.id for g, G in groups.items() for r, f in zip(G.records, is_flagged[g]) if f)
    with mo.status.spinner("split-half curves, one group at a time"):
        _out = curves(mode_pick.value, sigma_pick.value, repeats_pick.value, _key)
    curve_fig, _ax = plt.subplots(figsize=(7, 4))
    curve_table = []
    for _g, (_al, _sh) in _out.items():
        _ns = list(_al)
        _line = _ax.errorbar(_ns, [np.mean(_al[n]) for n in _ns], [np.std(_al[n]) for n in _ns], marker="o",
                             capsize=3, label=_g)
        _ax.plot(_ns, [np.mean(_sh[n]) for n in _ns], ls="--", color=_line[0].get_color(), alpha=0.6)
        curve_table += [{"group": _g, "N": n, "aligned mean": round(float(np.mean(_al[n])), 4),
                         "aligned std": round(float(np.std(_al[n])), 4), "control mean": round(float(np.mean(_sh[n])), 4)}
                        for n in _ns]
    _ax.set_xscale("log", base=2)
    _ax.set_xlabel("images per half (N)")
    _ax.set_ylabel("correlation of the two averages")
    _ax.set_title(f"black: split-half correlation vs N ({mode_pick.value} residuals)")
    _ax.axhline(0, color=CB["grey"], lw=0.5)
    _ax.legend()
    curve_fig.tight_layout()
    mo.hstack([curve_fig, mo.ui.table(curve_table, selection=None, page_size=30)], justify="start")
    return


@app.cell
def section_scores():
    mo.md(r"""
    ## 10 · Per-image score (focus group)

    Each image's full-size residual against its group's fingerprint, rebuilt **without that image**
    (leave-one-out) when the image is part of it.

    - **corr**: correlation with the fingerprint. **shared peaks**: of the image's top 20 with the fingerprint's top 20.
    """)
    return


@app.cell
def image_scores(
    flagged_pick,
    fp_sums,
    group_pick,
    groups,
    is_edge,
    is_flagged,
    mode_pick,
    sigma_pick,
    spectrum_view,
):
    _g = group_pick.value
    _G, _flag = groups[_g], is_flagged[_g]
    _total, _n = fp_sums[_g]
    _fp_pts = {(p["fx"], p["fy"]) for p in fl.find_peaks(spectrum_view(_total / _n)[0], 20, 7, False)}
    scores = []
    for _k in mo.status.progress_bar(range(len(_G.rows)), title=f"scoring {_g} images"):
        _r = fl.residuals(_G.pixels[[_G.rows[_k]]], mode_pick.value, TARGET, sigma_pick.value)[0]
        _ref = fl.leave_one_out(_total, _n, _r) if (flagged_pick.value or not _flag[_k]) else _total / _n
        _pts = {(p["fx"], p["fy"]) for p in fl.find_peaks(spectrum_view(_r)[0], 20, 7, False)}
        _rec = _G.records[_k]
        scores.append({"seed": _rec.seed, "id": _rec.id, "flagged": "yes" if _flag[_k] else "",
                       "edge case": "yes" if is_edge[_g][_k] else "",
                       "corr": round(fl.corr(_r, _ref), 4), "shared peaks": len(_pts & _fp_pts),
                       "residual std": round(float(_r.std()), 3)})
    _cl = [s for s in scores if not s["flagged"]]
    mo.vstack([mo.md(f"clean images: median shared peaks **{np.median([s['shared peaks'] for s in _cl]):.0f}/20** · "
                     f"median corr **{np.median([s['corr'] for s in _cl]):.4f}**"),
               mo.ui.table(sorted(scores, key=lambda s: -s["corr"]), selection=None, page_size=40,
                           label=f"{_g}: {len(scores)} images (sortable)")])
    return


@app.cell
def section_zoom():
    mo.md(r"""
    ## 11 · Pixel zoom

    The same crop of the focus group's fingerprint and one image, each pixel a visible square (stretched).
    """)
    return


@app.cell
def zoom_controls(group_pick, groups):
    _G = groups[group_pick.value]
    zx_pick = mo.ui.number(0, _G.size[0] - 1, value=0, label="x")
    zy_pick = mo.ui.number(0, _G.size[1] - 1, value=0, label="y")
    zsize_pick = mo.ui.dropdown({"16": 16, "32": 32, "64": 64}, value="32", label="crop size")
    zimage_pick = mo.ui.dropdown({f"{r.seed} · {r.id}": k for k, r in enumerate(_G.records)},
                                 value=f"{_G.records[0].seed} · {_G.records[0].id}", label="image", searchable=True)
    mo.hstack([zx_pick, zy_pick, zsize_pick, zimage_pick], justify="start")
    return zimage_pick, zsize_pick, zx_pick, zy_pick


@app.cell
def zoom_view(
    fps,
    group_pick,
    groups,
    k_pick,
    mode_pick,
    sigma_pick,
    zimage_pick,
    zsize_pick,
    zx_pick,
    zy_pick,
):
    _G, _size = groups[group_pick.value], zsize_pick.value
    _scale = max(1, 384 // _size)
    _x, _y = min(int(zx_pick.value), _G.size[0] - _size), min(int(zy_pick.value), _G.size[1] - _size)
    _r = fl.residuals(_G.pixels[[_G.rows[zimage_pick.value]]], mode_pick.value, TARGET, sigma_pick.value)[0]
    _panels = {"fingerprint": fps[group_pick.value], "image": _r}
    mo.hstack([mo.vstack([mo.md(f"**{name}** · x {_x}–{_x + _size - 1}, y {_y}–{_y + _size - 1}"),
                          mo.image(fl.png(_z := fl.zoom(fl.stretched(a, k_pick.value)[0], _x, _y, _size, _scale)),
                                   width=_z.shape[1])]) for name, a in _panels.items()], justify="start")
    return


if __name__ == "__main__":
    app.run()
