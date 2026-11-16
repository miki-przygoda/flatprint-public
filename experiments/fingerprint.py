import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def setup():
    import functools
    import json
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    import fingerprint_lib as fl
    from flatprint.dataset import load_matrix, read_ids
    from flatprint.prompts import rgb_of_prompt

    DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
    FLAGGED = DATA / "flat-rgb-flagged" / "flagged.md"
    # Okabe-Ito palette: stays distinct under the common colour-vision deficiencies. Heatmaps use magma
    # (sequential) and RdBu (diverging), which are colour-blind safe too.
    CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
          "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#999999", "black": "#000000"}
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[CB[k] for k in ("blue", "orange", "green", "vermillion",
                                                                          "purple", "sky", "black")])
    return CB, DATA, FLAGGED, datetime, fl, functools, json, load_matrix, mo, np, plt, read_ids, rgb_of_prompt


@app.cell
def intro(mo):
    mo.md(r"""
    # Fingerprint

    Works on the `flat-rgb` set (431 images, 12 colours) and the decoder-only reference `vae-flat-rgb`
    (the same 12 colours through Z-Image-Turbo's VAE alone, no diffusion).

    A **residual** is what is left of an image once its colour is taken away; a **fingerprint** is the
    average residual over many images. The controls below apply to every section.
    """)
    return


@app.cell
def load_data(DATA, FLAGGED, load_matrix, np, read_ids, rgb_of_prompt):
    matrix = load_matrix(DATA, "flat-rgb")
    flagged = read_ids(FLAGGED) if FLAGGED.exists() else set()
    is_flagged = np.array([r.id in flagged for r in matrix.records])
    colours = list(dict.fromkeys(matrix.names))
    target_of = {n: rgb_of_prompt(r.prompt) for n, r in zip(matrix.names, matrix.records)}
    rows_of = {c: [i for i, n in enumerate(matrix.names) if n == c] for c in colours}
    try:
        vae = load_matrix(DATA, "vae-flat-rgb")
        vae_row = {n: i for i, n in enumerate(vae.names)}
    except (ValueError, IndexError):
        vae, vae_row = None, {}
    return colours, is_flagged, matrix, rows_of, target_of, vae, vae_row


@app.cell
def controls(colours, mo):
    mode_pick = mo.ui.radio(["mean", "target", "highpass"], value="mean", inline=True, label="residual")
    sigma_pick = mo.ui.slider(1.0, 10.0, value=3.0, step=0.5, label="highpass blur σ (px)", show_value=True)
    flagged_pick = mo.ui.switch(value=False, label="include flagged images in fingerprints")
    colour_pick = mo.ui.dropdown(colours, value=colours[0], label="focus colour")
    k_pick = mo.ui.slider(1, 10, value=4, label="stretch ±k std", show_value=True)
    hann_pick = mo.ui.switch(value=True, label="Hann window")
    guides_pick = mo.ui.switch(value=True, label="1/8 + 1/16 guides")
    clip_pick = mo.ui.slider(90.0, 100.0, value=99.9, step=0.1, label="spectrum clip %", show_value=True)
    mo.vstack([
        mo.hstack([colour_pick, mode_pick, sigma_pick, flagged_pick], justify="start"),
        mo.hstack([k_pick, hann_pick, guides_pick, clip_pick], justify="start"),
        mo.md("**residual**: `mean` = minus the image's mean colour · `target` = minus the RGB the prompt asked for · "
              "`highpass` = minus a blurred copy (also removes large-scale shading; σ sets the blur)"),
    ])
    return clip_pick, colour_pick, flagged_pick, guides_pick, hann_pick, k_pick, mode_pick, sigma_pick


@app.cell
def fingerprint_cache(fl, functools, is_flagged, matrix, rows_of, target_of):
    @functools.lru_cache(maxsize=8)
    def fingerprints(mode, sigma, include_flagged):
        """Sum and count of residuals per colour; cached per setting."""
        out = {}
        for c, rows in rows_of.items():
            idx = [i for i in rows if include_flagged or not is_flagged[i]]
            out[c] = fl.fingerprint_sum(matrix.pixels, idx, mode, target_of[c], sigma)
        return out

    return (fingerprints,)


@app.cell
def compute_fingerprints(flagged_pick, fingerprints, mode_pick, mo, sigma_pick):
    with mo.status.spinner(f"fingerprints ({mode_pick.value} residuals) for all 12 colours"):
        fps = {c: s / n for c, (s, n) in fingerprints(mode_pick.value, sigma_pick.value, flagged_pick.value).items()}
        fp_sums = fingerprints(mode_pick.value, sigma_pick.value, flagged_pick.value)
    return fp_sums, fps


@app.cell
def focus(colour_pick, fl, is_flagged, matrix, mode_pick, np, rows_of, sigma_pick, target_of, vae, vae_row):
    focus_colour = colour_pick.value
    focus_rows = rows_of[focus_colour]
    focus_res = fl.residuals(matrix.pixels[focus_rows], mode_pick.value, target_of[focus_colour], sigma_pick.value)
    focus_flagged = np.array([is_flagged[i] for i in focus_rows])
    vae_res = (fl.residuals(vae.pixels[[vae_row[focus_colour]]], mode_pick.value, target_of[focus_colour],
                            sigma_pick.value)[0] if vae is not None and focus_colour in vae_row else None)
    return focus_colour, focus_flagged, focus_res, focus_rows, vae_res


@app.cell
def spectrum_fn(clip_pick, fl, guides_pick, hann_pick):
    def spectrum_view(img):
        spec = fl.log_spectrum(img, hann_pick.value)
        shown = fl.show_spectrum(spec, clip_pick.value)
        return spec, (fl.with_guides(shown) if guides_pick.value else shown)

    return (spectrum_view,)


@app.cell
def section_fingerprint(mo):
    mo.md(r"""
    ## 1 · Fingerprint and decoder-only reference
    """)
    return


@app.cell
def fingerprint_view(fl, focus_colour, fp_sums, fps, k_pick, mo, spectrum_view, vae_res):
    _fp = fps[focus_colour]
    _img, _s = fl.stretched(_fp, k_pick.value)
    fp_spec, _fp_spec_img = spectrum_view(_fp)
    fp_peaks = fl.find_peaks(fp_spec, 20, 7, False)
    _cols = [mo.vstack([mo.md(f"**fingerprint** · {focus_colour} · n={fp_sums[focus_colour][1]} · ±{_s:.2f} levels"),
                        mo.image(fl.png(_img), width=512), mo.image(fl.png(_fp_spec_img), width=512)])]
    vae_peaks = []
    if vae_res is not None:
        _vimg, _vs = fl.stretched(vae_res, k_pick.value)
        _vspec, _vspec_img = spectrum_view(vae_res)
        vae_peaks = fl.find_peaks(_vspec, 20, 7, False)
        _shared = len({(p["fx"], p["fy"]) for p in fp_peaks} & {(p["fx"], p["fy"]) for p in vae_peaks})
        _cols.append(mo.vstack([
            mo.md(f"**decoder only** · ±{_vs:.2f} levels · corr with fingerprint "
                  f"**{fl.corr(_fp, vae_res):.4f}** · shared top-20 peaks **{_shared}/20**"),
            mo.image(fl.png(_vimg), width=512), mo.image(fl.png(_vspec_img), width=512)]))
    else:
        _cols.append(mo.md("No `vae-flat-rgb` image for this colour."))
    mo.vstack([mo.hstack(_cols, justify="start"),
               mo.ui.tabs({"fingerprint peaks": mo.ui.table(fp_peaks, selection=None, page_size=20),
                           "decoder-only peaks": mo.ui.table(vae_peaks, selection=None, page_size=20)})])
    return fp_peaks, fp_spec


@app.cell
def section_curve(mo):
    mo.md(r"""
    ## 2 · Sample-size curve

    Two disjoint random sets of N images from the focus colour (flagged ones left out), each averaged, then
    correlated. Repeated for each N. **Shifted control**: the same with one average rolled by a random offset.
    If the curve is still climbing at the largest N, more images would sharpen the fingerprint.
    """)
    return


@app.cell
def curve_controls(mo):
    repeats_pick = mo.ui.slider(3, 30, value=10, label="repeats per N", show_value=True)
    repeats_pick
    return (repeats_pick,)


@app.cell
def sample_curve(CB, fl, focus_colour, focus_flagged, focus_res, mo, np, plt, repeats_pick):
    _res = focus_res[~focus_flagged]
    _sizes = [n for n in (1, 2, 4, 8, 16, 32, 64) if 2 * n <= len(_res)]
    _rng = np.random.default_rng(0)
    with mo.status.spinner("split-half curve"):
        curve = fl.split_half_curve(_res, _sizes, repeats_pick.value, _rng)
        _control = {}
        for _n in curve:
            _vals = []
            for _ in range(repeats_pick.value):
                _o = _rng.permutation(len(_res))
                _a, _b = _res[_o[:_n]].mean(0), _res[_o[_n:2 * _n]].mean(0)
                _dy, _dx = _rng.integers(16, _a.shape[0] - 16), _rng.integers(16, _a.shape[1] - 16)
                _vals.append(fl.corr(_a, np.roll(_b, (_dy, _dx), axis=(0, 1))))
            _control[_n] = _vals
    curve_fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ns = list(curve)
    _ax.errorbar(_ns, [np.mean(curve[n]) for n in _ns], [np.std(curve[n]) for n in _ns], marker="o", capsize=3,
                 color=CB["blue"], label="aligned")
    _ax.errorbar(_ns, [np.mean(_control[n]) for n in _ns], [np.std(_control[n]) for n in _ns], marker="s",
                 capsize=3, color=CB["orange"], label="shifted control")
    _ax.set_xscale("log", base=2)
    _ax.set_xlabel("images per half (N)")
    _ax.set_ylabel("correlation of the two averages")
    _ax.set_title(f"{focus_colour}: split-half correlation vs N")
    _ax.axhline(0, color="grey", lw=0.5)
    _ax.legend()
    curve_fig.tight_layout()
    curve_table = [{"N": n, "aligned mean": round(float(np.mean(curve[n])), 4), "aligned std": round(float(np.std(curve[n])), 4),
                    "control mean": round(float(np.mean(_control[n])), 4), "control std": round(float(np.std(_control[n])), 4)}
                   for n in _ns]
    mo.hstack([curve_fig, mo.ui.table(curve_table, selection=None)], justify="start")
    return curve_fig, curve_table


@app.cell
def section_scores(mo):
    mo.md(r"""
    ## 3 · Per-image score (focus colour)

    Each image's residual against the fingerprint. For images inside the fingerprint, the fingerprint is
    rebuilt **without that image** (leave-one-out). Flagged images are scored against the fingerprint
    as it stands (they are only in it if the switch above is on).

    - **corr**: correlation of the image's residual with the fingerprint.
    - **shared peaks**: how many of the image's top-20 spectral peaks sit at the same positions as the fingerprint's top 20.
    """)
    return


@app.cell
def image_scores(fl, focus_colour, focus_flagged, focus_res, focus_rows, fp_peaks, fp_sums, matrix, mo, np, spectrum_view):
    _total, _n = fp_sums[focus_colour]
    _fp_pts = {(p["fx"], p["fy"]) for p in fp_peaks}
    _in_fp = [(not f) or (_n == len(focus_rows)) for f in focus_flagged]
    scores = []
    for _k in mo.status.progress_bar(range(len(focus_rows)), title="scoring images"):
        _r = focus_res[_k]
        _ref = fl.leave_one_out(_total, _n, _r) if _in_fp[_k] else _total / _n
        _spec, _ = spectrum_view(_r)
        _pts = {(p["fx"], p["fy"]) for p in fl.find_peaks(_spec, 20, 7, False)}
        _rec = matrix.records[focus_rows[_k]]
        scores.append({"seed": _rec.seed, "id": _rec.id, "flagged": "yes" if focus_flagged[_k] else "",
                       "corr": round(fl.corr(_r, _ref), 4), "shared peaks": len(_pts & _fp_pts),
                       "residual std": round(float(_r.std()), 3)})
    mo.ui.table(sorted(scores, key=lambda s: -s["corr"]), selection=None, page_size=40,
                label=f"{focus_colour}: {len(scores)} images (sortable)")
    return (scores,)


@app.cell
def section_colours(mo):
    mo.md(r"""
    ## 4 · Colour vs colour

    How well each colour's fingerprint matches every other colour's (correlation), and the spectral peak
    positions that appear in the top 20 of several colours.
    """)
    return


@app.cell
def colour_matrix(colours, fl, fps, mo, np, plt):
    colour_corr = np.array([[fl.corr(fps[a], fps[b]) for b in colours] for a in colours])
    matrix_fig, _ax = plt.subplots(figsize=(7, 6))
    _im = _ax.imshow(colour_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    _ax.set_xticks(range(len(colours)), colours, rotation=60, ha="right")
    _ax.set_yticks(range(len(colours)), colours)
    for (_i, _j), _v in np.ndenumerate(colour_corr):
        _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=7)
    matrix_fig.colorbar(_im, ax=_ax, shrink=0.8)
    _ax.set_title("fingerprint correlation between colours")
    matrix_fig.tight_layout()
    matrix_fig
    return colour_corr, matrix_fig


@app.cell
def shared_controls(mo):
    min_colours_pick = mo.ui.slider(2, 12, value=6, label="appears in at least N colours", show_value=True)
    min_colours_pick
    return (min_colours_pick,)


@app.cell
def shared_peaks(colours, fl, fps, min_colours_pick, mo, spectrum_view):
    _where = {}
    for _c in colours:
        _spec, _ = spectrum_view(fps[_c])
        for _p in fl.find_peaks(_spec, 20, 7, False):
            _where.setdefault((_p["fx"], _p["fy"], _p["fx·8/W"], _p["fy·8/H"]), []).append(_c)
    shared = sorted(({"fx": k[0], "fy": k[1], "fx·8/W": k[2], "fy·8/H": k[3], "colours": len(v), "which": ", ".join(v)}
                     for k, v in _where.items() if len(v) >= min_colours_pick.value), key=lambda r: -r["colours"])
    mo.ui.table(shared, selection=None, page_size=30, label=f"{len(shared)} peak positions in ≥ {min_colours_pick.value} colours")
    return (shared,)


@app.cell
def section_tracker(mo):
    mo.md(r"""
    ## 5 · Peak tracker

    Every frequency on the 1/16 lattice (multiples of 1/16 of the width and height), with its local contrast
    in each colour's fingerprint. Rows are sorted by mean contrast across colours; the heatmap shows the same
    for the top 30. Below: the same top-30 positions in every single image of the focus colour.
    """)
    return


@app.cell
def tracker(colours, fl, focus_colour, focus_res, focus_rows, fps, matrix, mo, np, plt, spectrum_view):
    _h, _w = fps[colours[0]].shape[:2]
    lattice_pts = fl.lattice(_w, _h, div=16)
    _strength = np.array([fl.lattice_strength(spectrum_view(fps[c])[0], lattice_pts) for c in colours])
    _order = np.argsort(_strength.mean(axis=0))[::-1]
    tracker_rows = [{"fx·16/W": fx * 16 // _w, "fy·16/H": fy * 16 // _h, "mean": round(float(_strength[:, j].mean()), 3),
                     **{c: round(float(_strength[i, j]), 2) for i, c in enumerate(colours)}}
                    for j in _order for fx, fy in [lattice_pts[j]]]
    top_pts = [lattice_pts[j] for j in _order[:30]]
    _labels = [f"({fx * 16 // _w},{fy * 16 // _h})" for fx, fy in top_pts]
    _per_image = np.array([fl.lattice_strength(spectrum_view(focus_res[k])[0], top_pts) for k in range(len(focus_rows))])
    tracker_fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(12, 9))
    _i1 = _a1.imshow(_strength[:, _order[:30]], aspect="auto", cmap="magma")
    _a1.set_yticks(range(len(colours)), colours)
    _a1.set_xticks(range(30), _labels, rotation=90, fontsize=7)
    _a1.set_title("contrast at the top-30 lattice positions, per colour fingerprint (positions in 1/16 steps)")
    tracker_fig.colorbar(_i1, ax=_a1)
    _i2 = _a2.imshow(_per_image, aspect="auto", cmap="magma")
    _a2.set_yticks(range(len(focus_rows)), [str(matrix.records[i].seed) for i in focus_rows], fontsize=5)
    _a2.set_xticks(range(30), _labels, rotation=90, fontsize=7)
    _a2.set_title(f"same positions in every {focus_colour} image (rows = seeds)")
    tracker_fig.colorbar(_i2, ax=_a2)
    tracker_fig.tight_layout()
    mo.vstack([tracker_fig, mo.ui.table(tracker_rows, selection=None, page_size=20,
                                        label=f"{len(lattice_pts)} lattice positions, sorted by mean contrast")])
    return tracker_fig, tracker_rows


@app.cell
def section_outliers(mo):
    mo.md(r"""
    ## 6 · Outlier finder (all colours)

    Simple measures for every image, to sort and spot unusual ones. Uses mean-colour residuals regardless of
    the setting above, so it stays quick.

    - **off target**: distance of the image's mean colour from the requested RGB.
    - **noise**: standard deviation of the residual.
    - **top band**: brightness of the top 64 rows minus the whole image (grey levels).
    - **edge**: brightness of the outer 16-pixel border minus the inside.
    - **corr**: correlation with its colour's fingerprint (leave-one-out, mean residuals, flagged excluded).
    """)
    return


@app.cell
def outliers(fl, is_flagged, matrix, mo, np, rows_of, target_of):
    outlier_rows = []
    for _c, _rows in mo.status.progress_bar(list(rows_of.items()), title="measuring every image"):
        _res = fl.residuals(matrix.pixels[_rows], "mean")
        _clean = np.array([not is_flagged[i] for i in _rows])
        _total, _n = _res[_clean].astype(np.float64).sum(0), int(_clean.sum())
        for _k, _i in enumerate(_rows):
            _px = matrix.pixels[_i].astype(np.float32)
            _grey = _px.mean(axis=2)
            _inner = _grey[16:-16, 16:-16]
            _border = (_grey.sum() - _inner.sum()) / (_grey.size - _inner.size)
            _ref = fl.leave_one_out(_total, _n, _res[_k]) if _clean[_k] else _total / _n
            _rec = matrix.records[_i]
            outlier_rows.append({
                "colour": _c, "seed": _rec.seed, "id": _rec.id, "flagged": "yes" if is_flagged[_i] else "",
                "off target": round(float(np.linalg.norm(_px.mean(axis=(0, 1)) - np.array(target_of[_c]))), 2),
                "noise": round(float(_res[_k].std()), 3),
                "top band": round(float(_grey[:64].mean() - _grey.mean()), 3),
                "edge": round(float(_border - _inner.mean()), 3),
                "corr": round(fl.corr(_res[_k], _ref), 4)})
    mo.ui.table(outlier_rows, selection=None, page_size=30, label="every flat-rgb image (sort any column)")
    return (outlier_rows,)


@app.cell
def section_zoom(mo):
    mo.md(r"""
    ## 7 · Pixel zoom

    The same crop of the fingerprint, one image and the decoder-only residual, each pixel a visible square (all stretched).
    """)
    return


@app.cell
def zoom_controls(focus_rows, matrix, mo):
    zx_pick = mo.ui.number(0, 1023, value=0, label="x")
    zy_pick = mo.ui.number(0, 1023, value=0, label="y")
    zsize_pick = mo.ui.dropdown({"16": 16, "32": 32, "64": 64}, value="32", label="crop size")
    zimage_pick = mo.ui.dropdown({f"{matrix.records[i].seed} · {matrix.records[i].id}": k for k, i in enumerate(focus_rows)},
                                 value=f"{matrix.records[focus_rows[0]].seed} · {matrix.records[focus_rows[0]].id}",
                                 label="image", searchable=True)
    mo.hstack([zx_pick, zy_pick, zsize_pick, zimage_pick], justify="start")
    return zimage_pick, zsize_pick, zx_pick, zy_pick


@app.cell
def zoom_view(fl, focus_colour, focus_res, fps, k_pick, mo, vae_res, zimage_pick, zsize_pick, zx_pick, zy_pick):
    _size = zsize_pick.value
    _scale = max(1, 384 // _size)
    _x, _y = min(int(zx_pick.value), 1024 - _size), min(int(zy_pick.value), 1024 - _size)
    _panels = {"fingerprint": fps[focus_colour], "image": focus_res[zimage_pick.value]}
    if vae_res is not None:
        _panels["decoder only"] = vae_res
    zoom_imgs = {name: fl.zoom(fl.stretched(a, k_pick.value)[0], _x, _y, _size, _scale) for name, a in _panels.items()}
    mo.hstack([mo.vstack([mo.md(f"**{name}** · x {_x}–{_x + _size - 1}, y {_y}–{_y + _size - 1}"),
                          mo.image(fl.png(z), width=z.shape[1])]) for name, z in zoom_imgs.items()], justify="start")
    return (zoom_imgs,)


@app.cell
def section_export(mo):
    mo.md(r"""
    ## 8 · Export figures

    Saves the current views and tables, with the settings, to `data/figures/<time>-<colour>/`.
    """)
    return


@app.cell
def export_button(mo):
    export_pick = mo.ui.run_button(label="export current views")
    export_pick
    return (export_pick,)


@app.cell
def export(DATA, clip_pick, colour_corr, colours, curve_fig, curve_table, datetime, export_pick, fl, flagged_pick,
           focus_colour, fp_peaks, fp_spec, fps, guides_pick, hann_pick, json, k_pick, matrix_fig, mo, mode_pick,
           scores, shared, sigma_pick, tracker_fig, tracker_rows, zoom_imgs):
    mo.stop(not export_pick.value)
    import csv

    _out = DATA / "figures" / f"{datetime.now():%Y%m%d-%H%M%S}-{focus_colour.replace(' ', '-')}"
    _out.mkdir(parents=True, exist_ok=True)
    (_out / "fingerprint.png").write_bytes(fl.png(fl.stretched(fps[focus_colour], k_pick.value)[0]))
    _shown = fl.show_spectrum(fp_spec, clip_pick.value)
    (_out / "spectrum.png").write_bytes(fl.png(fl.with_guides(_shown) if guides_pick.value else _shown))
    for _name, _z in zoom_imgs.items():
        (_out / f"zoom-{_name.replace(' ', '-')}.png").write_bytes(fl.png(_z))
    curve_fig.savefig(_out / "sample-size-curve.png", dpi=200)
    matrix_fig.savefig(_out / "colour-matrix.png", dpi=200)
    tracker_fig.savefig(_out / "peak-tracker.png", dpi=200)

    def _csv(name, rows):
        if rows:
            with open(_out / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    _csv("fingerprint-peaks.csv", fp_peaks)
    _csv("image-scores.csv", scores)
    _csv("shared-peaks.csv", shared)
    _csv("peak-tracker.csv", tracker_rows)
    _csv("sample-size-curve.csv", curve_table)
    _csv("colour-matrix.csv", [{"colour": a, **{b: round(float(colour_corr[i, j]), 4) for j, b in enumerate(colours)}}
                               for i, a in enumerate(colours)])
    (_out / "settings.json").write_text(json.dumps({
        "focus colour": focus_colour, "residual": mode_pick.value, "sigma": sigma_pick.value,
        "include flagged": flagged_pick.value, "stretch k": k_pick.value, "hann": hann_pick.value,
        "guides": guides_pick.value, "clip": clip_pick.value}, indent=2), encoding="utf-8")
    mo.md(f"Saved to `{_out}`")
    return


if __name__ == "__main__":
    app.run()
