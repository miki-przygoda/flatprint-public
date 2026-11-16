import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from flatprint.dataset import load_groups, read_ids
    from flatprint.store import load_manifest

    DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
    FLAGGED = DATA / "flat-rgb-flagged" / "flagged.md"
    return DATA, FLAGGED, load_groups, load_manifest, mo, np, read_ids


@app.cell
def _(DATA, load_manifest, mo):
    sets = sorted({r.set for r in load_manifest(DATA)})
    set_pick = mo.ui.dropdown(sets, value="flat-rgb" if "flat-rgb" in sets else sets[0], label="set")
    flag_pick = mo.ui.radio(["all", "without flagged", "only flagged"], value="all", inline=True,
                            label="flagged (data/flat-rgb-flagged/flagged.md)")
    mo.hstack([set_pick, flag_pick], justify="start")
    return flag_pick, set_pick


@app.cell
def _(DATA, FLAGGED, flag_pick, load_groups, mo, read_ids, set_pick):
    flagged = read_ids(FLAGGED) if FLAGGED.exists() else set()
    filters = {"all": {}, "without flagged": {"exclude": flagged}, "only flagged": {"only": flagged}}
    with mo.status.spinner(f"loading {set_pick.value}"):
        groups = load_groups(DATA, set_pick.value, **filters[flag_pick.value])
    mo.ui.table(
        [{"colour": g.name, "target RGB": g.target_rgb, "size": f"{g.size[0]}x{g.size[1]}",
          "images": len(g.seeds)} for g in groups],
        selection=None, label=f"{set_pick.value}: {sum(len(g.seeds) for g in groups)} images in {len(groups)} groups",
    )
    return (groups,)


@app.cell
def _(groups, mo):
    group_pick = mo.ui.dropdown({f"{g.name} {g.size[0]}x{g.size[1]}": i for i, g in enumerate(groups)},
                                value=next(iter(f"{g.name} {g.size[0]}x{g.size[1]}" for g in groups)), label="group")
    group_pick
    return (group_pick,)


@app.cell
def _(group_pick, groups, mo):
    g = groups[group_pick.value]
    image_pick = mo.ui.slider(0, len(g.seeds) - 1, value=0, label="image", show_value=True)
    image_pick
    return g, image_pick


@app.cell
def _(DATA, g, image_pick, mo):
    from flatprint.store import resolve

    rec = g.records[image_pick.value]
    mo.vstack([
        mo.md(f"seed `{rec.seed}` · id `{rec.id}` · `{rec.file}`"),
        mo.image(str(resolve(DATA, rec)), width=512),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Analysis

    `groups` is a list of `Group`s, one per colour and size:
    `g.name`, `g.target_rgb` (what the prompt asked for), `g.size`, `g.seeds`, `g.records`,
    and `g.pixels`, a `uint8` array of shape `(n, height, width, 3)`.

    ### Special Images: - Unique
    1. seed 373246173 · id 852784b35510d0fe · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/852784b35510d0fe.png
    2. seed 686297256 · id 6acc47d1468422a4 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/6acc47d1468422a4.png
    3. This one is white and it shouldnt be? -- seed 1333669507 · id b3a59efe3ef9038c · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/b3a59efe3ef9038c.png
    4. seed 1453966580 · id 8f9176f5c41fbe20 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/8f9176f5c41fbe20.png
    5. seed 1580573816 · id 283781a0896ddcb9 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/283781a0896ddcb9.png
    6. seed 2030783831 · id d1f5bebafd3f32a5 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/d1f5bebafd3f32a5.png
    7. seed 2090447729 · id 22db358eaccbb05c · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/22db358eaccbb05c.png
    8. seed 1527420094 · id 107286ccdd850af4 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/107286ccdd850af4.png
    9. seed 1950823326 · id fa1f30a3d9aafede · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/fa1f30a3d9aafede.png
    10. seed 872154173 · id 793c26daa4ccac4b · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/793c26daa4ccac4b.png
    11. seed 1004988300 · id 2f208f9342631bad · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/2f208f9342631bad.png
    12. seed 1401164884 · id 0cc9fc612ee749b4 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/0cc9fc612ee749b4.png
    13. seed 986995544 · id d184498b0dbe916e · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/d184498b0dbe916e.png
    14. seed 1144694011 · id 12b2855743a22d4b · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/12b2855743a22d4b.png
    15. seed 1193464322 · id 6f455aeabd9ceffa · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/6f455aeabd9ceffa.png
    16. seed 1247384036 · id f938bc50a8e83508 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/f938bc50a8e83508.png
    17. seed 1315653312 · id 9b397775e3803491 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/9b397775e3803491.png
    18. seed 1574783764 · id 31e01e8e9e2e006b · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/31e01e8e9e2e006b.png
    19. seed 1601373167 · id 838ce310ed526f86 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/838ce310ed526f86.png
    20. seed 100819696 · id 6b19f0f29e5e45ff · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/6b19f0f29e5e45ff.png
    21. seed 779683399 · id 8b18d7d4b62ca491 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/8b18d7d4b62ca491.png
    22. seed 1045580737 · id 555caabd0dd38487 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/555caabd0dd38487.png
    23. seed 1937005193 · id 414848995b77f116 · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/414848995b77f116.png
    24. seed 2008855316 · id 4e8a2ac81de6313d · shards/macbook-pro/images/flat-rgb/Tongyi-MAI__Z-Image-Turbo/4e8a2ac81de6313d.png
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Breakdown and per-colour average images

    Uses `flat-rgb` without the flagged images (the switch above only affects the viewer).
    """)
    return


@app.cell
def _(DATA, FLAGGED, load_groups, mo, np, read_ids):
    _flagged = read_ids(FLAGGED) if FLAGGED.exists() else set()
    _all = load_groups(DATA, "flat-rgb")
    clean = load_groups(DATA, "flat-rgb", exclude=_flagged)
    _n_all = {g.name: len(g.seeds) for g in _all}
    _rows = []
    for _g in clean:
        _px = _g.pixels.astype(np.float32)
        _rows.append({"colour": _g.name, "target RGB": _g.target_rgb, "images": _n_all[_g.name],
                      "flagged": _n_all[_g.name] - len(_g.seeds), "used": len(_g.seeds),
                      "mean RGB": tuple(_px.mean((0, 1, 2)).round(1).tolist()),
                      "median per-image std": float(np.median(_px.std((1, 2)).mean(1)).round(2))})
    mo.ui.table(_rows, selection=None, label="flat-rgb, flagged images left out")
    return (clean,)


@app.cell
def _(clean, np):
    # The new image per colour: every pixel position averaged across that colour's images.
    averages = {g.name: g.pixels.astype(np.float64).mean(axis=0) for g in clean}
    return (averages,)


@app.cell
def _(mo):
    stretch_k = mo.ui.slider(1, 10, value=4, label="stretch: ±k std of the average image", show_value=True)
    stretch_k
    return (stretch_k,)


@app.cell
def _(np):
    import io as _io

    from PIL import Image as _Image

    def stretched(avg, k):
        """Deviation from the image's own mean per channel, scaled so ±k std spans 0..255 (grey = average)."""
        dev = avg - avg.mean(axis=(0, 1))
        s = k * dev.std()
        return np.clip(128 + 127 * dev / s, 0, 255).astype(np.uint8), s

    def png(arr):
        b = _io.BytesIO()
        _Image.fromarray(arr).save(b, "PNG")
        return b.getvalue()

    return png, stretched


@app.cell
def _(averages, clean, mo, np, png, stretch_k, stretched):
    _n = {g.name: len(g.seeds) for g in clean}
    _tiles = []
    for _name, _avg in averages.items():
        _img, _s = stretched(_avg, stretch_k.value)
        _tiles.append(mo.vstack([
            mo.md(f"**{_name}** · n={_n[_name]} · ±{_s:.2f} levels"),
            mo.hstack([mo.image(png(np.round(_avg).astype(np.uint8)), width=150),
                       mo.image(png(_img), width=150)]),
        ]))
    mo.vstack([mo.md("Left: the average as it is. Right: stretched."),
               mo.hstack(_tiles[:4]), mo.hstack(_tiles[4:8]), mo.hstack(_tiles[8:])])
    return


@app.cell
def _(averages, mo):
    avg_pick = mo.ui.dropdown(list(averages), value=next(iter(averages)), label="full size")
    avg_pick
    return (avg_pick,)


@app.cell
def _(avg_pick, averages, mo, png, stretch_k, stretched):
    _img, _s = stretched(averages[avg_pick.value], stretch_k.value)
    mo.vstack([mo.md(f"**{avg_pick.value}**, stretched ±{_s:.2f} levels, 1024×1024 at 1:1"), mo.image(png(_img), width=1024)])
    return


if __name__ == "__main__":
    app.run()
