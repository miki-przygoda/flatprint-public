import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def setup():
    import io
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    import cleanup_lib as cl
    from flatprint.dataset import load_matrix, read_ids
    from flatprint.store import load_manifest

    DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))

    def flagged_path(set_name):
        return DATA / f"{set_name}-flagged" / "flagged.md"

    return DATA, Image, cl, datetime, flagged_path, io, load_manifest, load_matrix, mo, np, read_ids


@app.cell
def intro(mo):
    mo.md(r"""
    # Cleanup

    Finds the images in a set that don't look like the rest, so only those need checking by hand.

    1. Pick the set to clean and a **reference** set whose clean images show what normal looks like (matched
       by colour name; the reference set's own flagged images are left out).
    2. Every image is shrunk to a small thumbnail (so any size compares with any other) and scored against
       the average reference thumbnail of its colour:
       - **brightness**: difference in overall level (white or grey instead of black, etc.)
       - **shape**: difference once both have their mean removed (frames, N-gons, gradients)
       - **spot**: the largest single-point difference (small marks, text)
    3. Thresholds come from the reference itself: each reference image scored against the others, the worst
       of those × the margin (never below the floor).
    4. Look through the **to check** gallery, tick the real outliers in the table, and append them to
       `data/<set>-flagged/flagged.md`, the file every other notebook reads.
    """)
    return


@app.cell
def pickers(DATA, load_manifest, mo):
    _sets = sorted({r.set for r in load_manifest(DATA)})
    target_pick = mo.ui.dropdown(_sets, value="flat-black-sizes" if "flat-black-sizes" in _sets else _sets[0],
                                 label="set to clean")
    ref_pick = mo.ui.dropdown(_sets, value="flat-rgb" if "flat-rgb" in _sets else _sets[0], label="reference set")
    thumb_pick = mo.ui.dropdown({"64": 64, "128": 128, "256": 256}, value="128", label="thumbnail size")
    reload_pick = mo.ui.button(value=0, on_click=lambda v: v + 1, label="reload flagged lists")
    mo.hstack([target_pick, ref_pick, thumb_pick, reload_pick], justify="start")
    return ref_pick, reload_pick, target_pick, thumb_pick


@app.cell
def thumbs_fn(DATA, cl, load_manifest, load_matrix, mo, np):
    def load_thumbs(set_name, n):
        """Every image of a set: (mats by size, rows [(size, row)], records, colour names, thumbnails (N, n, n, 3))."""
        sizes = sorted({(r.width, r.height) for r in load_manifest(DATA) if r.set == set_name})
        mats = {s: load_matrix(DATA, set_name, size=s) for s in sizes}
        rows = [(s, i) for s in sizes for i in range(len(mats[s].records))]
        thumbs = [cl.thumbnail(mats[s].pixels[i], n)
                  for s, i in mo.status.progress_bar(rows, title=f"thumbnails for {set_name}")]
        return (mats, rows, [mats[s].records[i] for s, i in rows], [mats[s].names[i] for s, i in rows],
                np.stack(thumbs) if thumbs else None)

    return (load_thumbs,)


@app.cell
def load_target(load_thumbs, target_pick, thumb_pick):
    t_mats, t_rows, t_recs, t_names, t_thumbs = load_thumbs(target_pick.value, thumb_pick.value)
    return t_mats, t_names, t_recs, t_rows, t_thumbs


@app.cell
def load_reference(load_thumbs, ref_pick, thumb_pick):
    _, _, r_recs, r_names, r_thumbs = load_thumbs(ref_pick.value, thumb_pick.value)
    return r_names, r_recs, r_thumbs


@app.cell
def flagged_lists(flagged_path, mo, read_ids, ref_pick, reload_pick, target_pick):
    reload_pick.value
    _tp, _rp = flagged_path(target_pick.value), flagged_path(ref_pick.value)
    t_flagged = read_ids(_tp) if _tp.exists() else set()
    r_flagged = read_ids(_rp) if _rp.exists() else set()
    mo.md(f"`{_tp.parent.name}/flagged.md`: **{len(t_flagged)}** flagged · "
          f"`{_rp.parent.name}/flagged.md`: **{len(r_flagged)}** flagged (left out of the reference)")
    return r_flagged, t_flagged


@app.cell
def threshold_controls(mo):
    margin_pick = mo.ui.slider(1.0, 5.0, value=1.5, step=0.1, label="margin × worst reference score", show_value=True)
    floor_picks = mo.ui.dictionary({
        "brightness": mo.ui.number(0, 255, value=0, step=0.5, label="brightness floor"),
        "shape": mo.ui.number(0, 255, value=0, step=0.5, label="shape floor"),
        "spot": mo.ui.number(0, 255, value=0, step=0.5, label="spot floor"),
    })
    mo.vstack([mo.hstack([margin_pick, *floor_picks.values()], justify="start"),
               mo.md("Raise a floor if a whole group lands in *to check* for one score while looking fine.")])
    return floor_picks, margin_pick


@app.cell
def calibration(cl, floor_picks, margin_pick, mo, np, r_flagged, r_names, r_recs, r_thumbs):
    refs = {}
    for _c in dict.fromkeys(r_names):
        _idx = [i for i, (n, r) in enumerate(zip(r_names, r_recs)) if n == _c and r.id not in r_flagged]
        if len(_idx) < 3:
            continue
        _cal = cl.calibrate(r_thumbs[_idx])
        refs[_c] = {"mean": r_thumbs[_idx].mean(axis=0), "n": len(_idx),
                    "limit": {k: max(margin_pick.value * float(_cal[:, j].max()), float(floor_picks.value[k]))
                              for j, k in enumerate(cl.SCORES)},
                    "worst": {k: float(_cal[:, j].max()) for j, k in enumerate(cl.SCORES)},
                    "p95": {k: float(np.percentile(_cal[:, j], 95)) for j, k in enumerate(cl.SCORES)}}
    mo.ui.table([{"colour": c, "reference images": v["n"],
                  **{f"{k} worst": round(v["worst"][k], 2) for k in cl.SCORES},
                  **{f"{k} p95": round(v["p95"][k], 2) for k in cl.SCORES},
                  **{f"{k} threshold": round(v["limit"][k], 2) for k in cl.SCORES}} for c, v in refs.items()],
                selection=None, label="reference per colour (each image scored against the others)")
    return (refs,)


@app.cell
def scoring(cl, refs, t_flagged, t_names, t_recs, t_rows, t_thumbs):
    scored = []
    for _k, ((_s, _i), _rec, _c) in enumerate(zip(t_rows, t_recs, t_names)):
        _row = {"k": _k, "size": f"{_s[0]}x{_s[1]}", "colour": _c, "seed": _rec.seed, "id": _rec.id,
                "flagged": "yes" if _rec.id in t_flagged else ""}
        if _c not in refs:
            scored.append({**_row, **{k: None for k in cl.SCORES}, "reasons": "no reference"})
            continue
        _sc = cl.scores(t_thumbs[_k], refs[_c]["mean"])
        _over = [k for k in cl.SCORES if _sc[k] > refs[_c]["limit"][k]]
        scored.append({**_row, **{k: round(v, 2) for k, v in _sc.items()}, "reasons": ", ".join(_over)})
    return (scored,)


@app.cell
def summary(mo, scored):
    _groups = {}
    for _r in scored:
        _g = _groups.setdefault((_r["colour"], _r["size"]), {"colour": _r["colour"], "size": _r["size"], "images": 0,
                                                            "already flagged": 0, "to check": 0, "brightness": 0,
                                                            "shape": 0, "spot": 0, "no reference": 0})
        _g["images"] += 1
        if _r["flagged"]:
            _g["already flagged"] += 1
        elif _r["reasons"]:
            _g["to check"] += 1
            for _reason in _r["reasons"].split(", "):
                _g[_reason] += 1
    _total = sum(g["to check"] for g in _groups.values())
    mo.ui.table(list(_groups.values()), selection=None,
                label=f"{len(scored)} images · {_total} to check (not already flagged); reason columns overlap")
    return


@app.cell
def section_gallery(mo):
    mo.md(r"""
    ## Gallery

    *to check*: images over a threshold that aren't flagged yet. *passed*: the rest, to spot anything missed.
    Tiles are the images shrunk for display; open one at full size in `flat_images.py` or `resolutions.py`.
    """)
    return


@app.cell
def gallery_controls(mo, scored):
    show_pick = mo.ui.radio(["to check", "passed", "already flagged", "all"], value="to check", inline=True,
                            label="show")
    order_pick = mo.ui.dropdown(["size, seed", "brightness", "shape", "spot"], value="size, seed", label="order by")
    tile_pick = mo.ui.slider(96, 320, value=192, step=32, label="tile px", show_value=True)
    mo.hstack([show_pick, order_pick, tile_pick], justify="start")
    return order_pick, show_pick, tile_pick


@app.cell
def shown_list(order_pick, scored, show_pick):
    _keep = {"to check": lambda r: r["reasons"] and not r["flagged"],
             "passed": lambda r: not r["reasons"] and not r["flagged"],
             "already flagged": lambda r: r["flagged"],
             "all": lambda r: True}[show_pick.value]
    shown = [r for r in scored if _keep(r)]
    if order_pick.value != "size, seed":
        shown.sort(key=lambda r: -(r[order_pick.value] or 0))
    return (shown,)


@app.cell
def page_control(mo, shown):
    page_pick = mo.ui.slider(1, max(1, -(-len(shown) // 40)), value=1, label=f"page (40 per page, {len(shown)} images)",
                             show_value=True)
    page_pick
    return (page_pick,)


@app.cell
def gallery(Image, io, mo, page_pick, shown, t_mats, t_rows, tile_pick):
    def _tile(r):
        _s, _i = t_rows[r["k"]]
        _im = Image.fromarray(t_mats[_s].pixels[_i]).resize((tile_pick.value, tile_pick.value), Image.LANCZOS)
        _b = io.BytesIO()
        _im.save(_b, "PNG")
        return mo.vstack([mo.image(_b.getvalue(), width=tile_pick.value),
                          mo.md(f"**{r['k']}** · {r['size']} · {r['seed']}  \n{r['reasons'] or 'passed'}")], gap=0.1)

    _page = shown[(page_pick.value - 1) * 40: page_pick.value * 40]
    mo.hstack([_tile(r) for r in _page], wrap=True, justify="start", gap=0.6) if _page else mo.md("Nothing to show.")
    return


@app.cell
def section_flag(mo):
    mo.md(r"""
    ## Flag

    Tick the real outliers (the `k` number matches the gallery), add a note if you like, then append them.
    Ids already in the file are skipped; existing notes are kept. Press **reload flagged lists** afterwards.
    """)
    return


@app.cell
def flag_table(mo, shown):
    pick_table = mo.ui.table(shown, selection="multi", page_size=40, label="tick the ones to flag")
    note_pick = mo.ui.text(placeholder="optional note, e.g. white frame / text", full_width=True, label="note")
    append_pick = mo.ui.run_button(label="append ticked to flagged.md")
    mo.vstack([pick_table, mo.hstack([note_pick, append_pick])])
    return append_pick, note_pick, pick_table


@app.cell
def append_flagged(append_pick, datetime, flagged_path, mo, note_pick, pick_table, read_ids, target_pick):
    mo.stop(not append_pick.value)
    _path = flagged_path(target_pick.value)
    _have = read_ids(_path) if _path.exists() else set()
    _new = [r for r in pick_table.value if r["id"] not in _have]
    mo.stop(not _new, mo.md("Nothing new to add."))
    _path.parent.mkdir(parents=True, exist_ok=True)
    _note = f" · {note_pick.value.strip()}" if note_pick.value.strip() else ""
    _text = "" if _path.exists() else f"# Flagged images: {target_pick.value}\n"
    _text += f"\n## {datetime.now():%Y-%m-%d %H:%M} · cleanup.py\n\n"
    _text += "".join(f"- {r['size']} · seed {r['seed']} · id {r['id']} · {r['reasons'] or 'passed'}{_note}\n"
                     for r in _new)
    with open(_path, "a", encoding="utf-8") as _f:
        _f.write(_text)
    mo.md(f"Added **{len(_new)}** to `{_path}`. Press **reload flagged lists** to update.")
    return


if __name__ == "__main__":
    app.run()
