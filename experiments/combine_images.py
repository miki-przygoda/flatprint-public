import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from PIL import Image

    from flatprint.dataset import load_matrix, read_ids
    from flatprint.store import load_manifest

    DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
    FLAGGED = DATA / "flat-rgb-flagged" / "flagged.md"
    return (
        DATA,
        FLAGGED,
        Image,
        io,
        load_manifest,
        load_matrix,
        mo,
        np,
        read_ids,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Combine images

    Pick a set, tick the images to combine, and see every pixel position averaged across them.
    Flagged images (`data/flat-rgb-flagged/flagged.md`) are greyed out and not ticked, but you can still tick them.
    Each set is held as one `(n, height, width, 3)` uint8 matrix (`matrix.pixels`, cached in `data/cache/`).
    """)
    return


@app.cell
def set_picker(DATA, load_manifest, mo):
    _sizes = {}
    for _r in load_manifest(DATA):
        _sizes.setdefault(_r.set, set()).add((_r.width, _r.height))
    _options = {f"{s} {w}x{h}": (s, (w, h)) for s in sorted(_sizes) for w, h in sorted(_sizes[s])}
    set_pick = mo.ui.dropdown(_options, value=next((k for k in _options if k.startswith("flat-rgb ")), next(iter(_options))),
                              label="set")
    set_pick
    return (set_pick,)


@app.cell
def load_set(DATA, FLAGGED, load_matrix, mo, np, read_ids, set_pick):
    _set, _size = set_pick.value
    with mo.status.spinner(f"loading {_set} (first time builds the cache)"):
        matrix = load_matrix(DATA, _set, size=_size)
        means = matrix.pixels.mean(axis=(1, 2))           # (n, 3) mean RGB per image
    flagged = read_ids(FLAGGED) if FLAGGED.exists() else set()
    is_flagged = np.array([r.id in flagged for r in matrix.records])
    return flagged, is_flagged, matrix, means


@app.cell
def colour_picker(matrix, mo):
    colour_pick = mo.ui.multiselect(list(dict.fromkeys(matrix.names)), value=list(dict.fromkeys(matrix.names))[:1],
                                    label="colours to list")
    colour_pick
    return (colour_pick,)


@app.cell
def image_table(colour_pick, is_flagged, matrix, means, mo):
    rows = [i for i, n in enumerate(matrix.names) if n in colour_pick.value]
    _data = [{"row": i, "colour": matrix.names[i], "seed": matrix.records[i].seed, "id": matrix.records[i].id,
              "flagged": "yes" if is_flagged[i] else "",
              "mean RGB": ", ".join(f"{v:.1f}" for v in means[i])} for i in rows]

    def _grey_flagged(row_id, _column, _value):
        return {"color": "#9ca3af", "fontStyle": "italic"} if is_flagged[rows[int(row_id)]] else {}

    table = mo.ui.table(_data, selection="multi", page_size=50, style_cell=_grey_flagged,
                        initial_selection=[k for k, i in enumerate(rows) if not is_flagged[i]],
                        label=f"{len(rows)} images; flagged ones are grey and start unticked")
    table
    return (table,)


@app.cell
def _(matrix, np):
    def average(idx, chunk=32):
        """Mean over the chosen images at every pixel position, summed in chunks to keep memory low."""
        total = np.zeros(matrix.pixels.shape[1:], dtype=np.float64)
        for s in range(0, len(idx), chunk):
            total += matrix.pixels[idx[s:s + chunk]].sum(axis=0, dtype=np.float64)
        return total / len(idx)

    return (average,)


@app.cell
def combine(average, mo, table):
    chosen = sorted(r["row"] for r in table.value)
    mo.stop(not chosen, mo.md("Tick at least one image."))
    with mo.status.spinner(f"averaging {len(chosen)} images"):
        combined = average(chosen)
    return chosen, combined


@app.cell
def _(mo):
    stretch_k = mo.ui.slider(1, 10, value=4, label="stretch: ±k std", show_value=True)
    stretch_k
    return (stretch_k,)


@app.cell
def _(Image, io, np):
    def stretched(avg, k):
        """Deviation from the image's own mean per channel, scaled so ±k std spans 0..255 (grey = average)."""
        dev = avg - avg.mean(axis=(0, 1))
        s = k * dev.std()
        return np.clip(128 + 127 * dev / s, 0, 255).astype(np.uint8), s

    def png(arr):
        b = io.BytesIO()
        Image.fromarray(arr).save(b, "PNG")
        return b.getvalue()

    return png, stretched


@app.cell
def combined_view(chosen, combined, flagged, matrix, mo, np, png, stretch_k, stretched):
    _img, _s = stretched(combined, stretch_k.value)
    _colours = sorted({matrix.names[i] for i in chosen})
    _n_flagged = sum(matrix.records[i].id in flagged for i in chosen)
    _h, _w = combined.shape[:2]
    mo.vstack([
        mo.md(f"**{len(chosen)} images** ({_n_flagged} flagged) · {', '.join(_colours)} · "
              f"mean RGB {', '.join(f'{v:.1f}' for v in combined.mean(axis=(0, 1)))} · stretch ±{_s:.2f} levels"),
        mo.hstack([
            mo.vstack([mo.md("average"), mo.image(png(np.round(combined).astype(np.uint8)), width=_w // 2)]),
            mo.vstack([mo.md("stretched"), mo.image(png(_img), width=_w // 2)]),
        ], justify="start"),
        mo.md("stretched, 1:1"),
        mo.image(png(_img), width=_w),
    ])
    return


@app.cell
def _(np):
    def centred(a):
        """Remove each channel's own mean, so only the pattern is left."""
        return a - a.mean(axis=(0, 1))

    def corr(a, b):
        """Pearson correlation of two mean-removed arrays, over every pixel and channel."""
        a, b = a.ravel(), b.ravel()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    return centred, corr


@app.cell
def _(mo):
    mo.md(r"""
    ## Split-half test

    The ticked images are split at random into two halves, each half is averaged, and the two averages are
    correlated (each with its own mean colour removed). Repeated over many random splits.

    - **aligned**: the two half-averages compared pixel for pixel.
    - **shifted control**: the same, but one half-average is rolled by a random offset first, so positions no
      longer line up while the texture stays the same. This is what "no shared pattern" looks like.
    """)
    return


@app.cell
def _(mo):
    n_splits = mo.ui.slider(5, 100, value=20, step=5, label="random splits", show_value=True)
    n_splits
    return (n_splits,)


@app.cell
def split_half(average, centred, chosen, combined, corr, mo, n_splits, np):
    mo.stop(len(chosen) < 4, mo.md("The split-half test needs at least 4 ticked images."))
    _rng = np.random.default_rng(0)
    _total = combined * len(chosen)
    _aligned, _shifted, _channels = [], [], []
    for _ in mo.status.progress_bar(range(n_splits.value), title="splitting"):
        _order = _rng.permutation(chosen)
        _a_idx = sorted(_order[: len(chosen) // 2].tolist())
        half_a = average(_a_idx)
        half_b = (_total - half_a * len(_a_idx)) / (len(chosen) - len(_a_idx))
        _ca, _cb = centred(half_a), centred(half_b)
        _aligned.append(corr(_ca, _cb))
        _channels.append([corr(_ca[..., c], _cb[..., c]) for c in range(3)])
        _dy, _dx = _rng.integers(16, _ca.shape[0] - 16), _rng.integers(16, _ca.shape[1] - 16)
        _shifted.append(corr(_ca, np.roll(_cb, (_dy, _dx), axis=(0, 1))))
    split_aligned, split_shifted = np.array(_aligned), np.array(_shifted)
    _ch = np.array(_channels).mean(axis=0)
    mo.md(f"""
    **{len(chosen)} images**, halves of {len(chosen) // 2} and {len(chosen) - len(chosen) // 2}, {n_splits.value} splits

    | | mean correlation | std | min | max |
    |---|---|---|---|---|
    | aligned | {split_aligned.mean():.4f} | {split_aligned.std():.4f} | {split_aligned.min():.4f} | {split_aligned.max():.4f} |
    | shifted control | {split_shifted.mean():.4f} | {split_shifted.std():.4f} | {split_shifted.min():.4f} | {split_shifted.max():.4f} |

    Aligned, per channel (mean over splits): R {_ch[0]:.4f} · G {_ch[1]:.4f} · B {_ch[2]:.4f}
    """)
    return half_a, half_b, split_aligned, split_shifted


@app.cell
def split_half_view(half_a, half_b, mo, png, stretch_k, stretched):
    _a, _sa = stretched(half_a, stretch_k.value)
    _b, _sb = stretched(half_b, stretch_k.value)
    mo.vstack([
        mo.md("The two half-averages from the last split, stretched (same k as above), 1:1 top-left 256×256 crop"),
        mo.hstack([
            mo.vstack([mo.md(f"half A · ±{_sa:.2f} levels"), mo.image(png(_a[:256, :256]), width=512)]),
            mo.vstack([mo.md(f"half B · ±{_sb:.2f} levels"), mo.image(png(_b[:256, :256]), width=512)]),
        ], justify="start"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spectrum (2D FFT)

    Log magnitude of the 2D FFT of the mean-removed image (channels averaged to one grey channel first).
    The centre is zero frequency; the edges are the highest frequencies (a pattern that flips every pixel).
    Shown next to the spectrum of one single image of your choice.
    **Hann window**: fades the image to zero at its edges before the FFT, so a mismatch between opposite
    edges (which shows up as a bright cross through the centre) is removed.
    """)
    return


@app.cell
def spectrum_controls(matrix, mo):
    _labels = {f"{matrix.names[i]} · seed {r.seed} · {r.id}": i for i, r in enumerate(matrix.records)}
    _default = next((k for k in _labels if k.endswith("22db358eaccbb05c")), next(iter(_labels)))
    single_pick = mo.ui.dropdown(_labels, value=_default, label="single image to compare", searchable=True)
    clip_pick = mo.ui.slider(90.0, 100.0, value=99.9, step=0.1, label="display clip percentile", show_value=True)
    hann_pick = mo.ui.switch(value=False, label="Hann window")
    mo.hstack([single_pick, clip_pick, hann_pick], justify="start")
    return clip_pick, hann_pick, single_pick


@app.cell
def _(centred, np):
    def log_spectrum(img, hann=False):
        """Centred log-magnitude spectrum of an (h, w, 3) image, channels averaged to grey first.
        With hann=True the grey image is multiplied by a 2D Hann window before the FFT."""
        grey = centred(np.asarray(img, dtype=np.float64)).mean(axis=2)
        if hann:
            grey = grey * np.outer(np.hanning(grey.shape[0]), np.hanning(grey.shape[1]))
        return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(grey))))

    def show_spectrum(spec, clip):
        lo, hi = np.percentile(spec, [100 - clip, clip])
        return (np.clip((spec - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    return log_spectrum, show_spectrum


@app.cell
def spectra(chosen, clip_pick, combined, hann_pick, log_spectrum, matrix, mo, png, show_spectrum, single_pick):
    spectrum_avg = log_spectrum(combined, hann_pick.value)
    spectrum_single = log_spectrum(matrix.pixels[single_pick.value], hann_pick.value)
    _r = matrix.records[single_pick.value]
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md(f"average of {len(chosen)} ticked images"),
                       mo.image(png(show_spectrum(spectrum_avg, clip_pick.value)), width=512)]),
            mo.vstack([mo.md(f"{matrix.names[single_pick.value]} · seed {_r.seed} · {_r.id}"),
                       mo.image(png(show_spectrum(spectrum_single, clip_pick.value)), width=512)]),
        ], justify="start"),
        mo.md("Full size (1:1), average:"),
        mo.image(png(show_spectrum(spectrum_avg, clip_pick.value)), width=spectrum_avg.shape[1]),
        mo.md("Full size (1:1), single image:"),
        mo.image(png(show_spectrum(spectrum_single, clip_pick.value)), width=spectrum_single.shape[1]),
    ])
    return spectrum_avg, spectrum_single


@app.cell
def _(mo):
    mo.md(r"""
    ## Peak list

    The strongest points in each spectrum (same Hann setting as above). A peak is a point that is the
    largest in its 5×5 neighbourhood. Points within the centre radius are skipped, and of each mirrored
    pair (a real image's spectrum is symmetric through the centre) only one is listed.

    - **fx, fy**: frequency in cycles across the image (0 = no change, ±W/2 = flips every pixel).
    - **fx·8/W, fy·8/W**: the same as a fraction of the width in eighths (1.0 = 1/8 of the width).
    - **period x, y**: how many pixels the pattern takes to repeat along that axis (∞ = no change).
    - **log mag**: the value in the spectrum. **contrast**: log mag minus the average in a 31×31 area around
      it, i.e. how much it stands out from its surroundings.
    """)
    return


@app.cell
def peak_controls(mo):
    n_peaks = mo.ui.slider(5, 50, value=20, step=5, label="peaks", show_value=True)
    dc_radius = mo.ui.slider(0, 32, value=4, label="skip centre radius (bins)", show_value=True)
    skip_axes = mo.ui.switch(value=False, label="skip the centre cross (fx = 0 or fy = 0)")
    rank_by = mo.ui.radio(["contrast", "log mag"], value="contrast", inline=True, label="rank by")
    mo.hstack([n_peaks, dc_radius, skip_axes, rank_by], justify="start")
    return dc_radius, n_peaks, rank_by, skip_axes


@app.cell
def _(np):
    def find_peaks(spec, n, radius, skip_axes, rank_by):
        """Local maxima of a centred spectrum, one of each mirrored pair, ranked; returns table rows."""
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        padded = np.pad(spec, 2, mode="edge")
        local_max = np.lib.stride_tricks.sliding_window_view(padded, (5, 5)).max(axis=(2, 3))
        # mean over a 31x31 box via a summed-area table
        r = 15
        sat = np.pad(np.pad(spec, r, mode="edge"), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        k = 2 * r + 1
        box = (sat[k:, k:] - sat[:-k, k:] - sat[k:, :-k] + sat[:-k, :-k]) / (k * k)
        contrast = spec - box
        fy, fx = np.mgrid[0:h, 0:w]
        fy, fx = fy - cy, fx - cx
        keep = (spec == local_max) & (fx**2 + fy**2 > radius**2)
        keep &= (fy > 0) | ((fy == 0) & (fx > 0))          # one of each mirrored pair
        if skip_axes:
            keep &= (fx != 0) & (fy != 0)
        score = contrast if rank_by == "contrast" else spec
        ys, xs = np.nonzero(keep)
        order = np.argsort(score[ys, xs])[::-1][:n]
        rows = []
        for rank, j in enumerate(order, 1):
            y, x = ys[j], xs[j]
            rows.append({"rank": rank, "fx": int(fx[y, x]), "fy": int(fy[y, x]),
                         "fx·8/W": round(fx[y, x] * 8 / w, 3), "fy·8/H": round(fy[y, x] * 8 / h, 3),
                         "period x (px)": round(w / abs(fx[y, x]), 2) if fx[y, x] else "∞",
                         "period y (px)": round(h / abs(fy[y, x]), 2) if fy[y, x] else "∞",
                         "log mag": round(float(spec[y, x]), 3), "contrast": round(float(contrast[y, x]), 3)})
        return rows

    return (find_peaks,)


@app.cell
def peaks(dc_radius, find_peaks, matrix, mo, n_peaks, rank_by, single_pick, skip_axes, spectrum_avg, spectrum_single):
    _args = (n_peaks.value, dc_radius.value, skip_axes.value, rank_by.value)
    peaks_avg = find_peaks(spectrum_avg, *_args)
    peaks_single = find_peaks(spectrum_single, *_args)
    _r = matrix.records[single_pick.value]
    mo.ui.tabs({
        "average": mo.ui.table(peaks_avg, selection=None, page_size=50),
        f"single · {matrix.names[single_pick.value]} · {_r.id}": mo.ui.table(peaks_single, selection=None, page_size=50),
    })
    return peaks_avg, peaks_single


if __name__ == "__main__":
    app.run()
