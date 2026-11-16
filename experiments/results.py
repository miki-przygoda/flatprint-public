import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import functools
    import io
    import json
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    import fingerprint_lib as fl
    from flatprint.store import load_manifest, resolve

    DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
    RESULTS = Path(os.environ.get("FLATPRINT_RESULTS", DATA.parent / "logs" / "results"))
    THUMB = 220
    PREVIEW_MAX = 2048             # previews: whole picture downsampled above this long side; actual pixels = this crop
    CARD_W, CARD_H = 240, 370      # every card the same size, so a page reads as a grid
    TITLE = {"real": "Real Photo", "social media": "Social Media", "fingerprint test": "Z-Image",
             "other generated": "Z-Image", "unlabelled": "Maybe AI", "AI, other generator": "AI (other generator)",
             "robustness test": "Robustness test", "FLUX (same VAE)": "FLUX (same VAE)"}
    REAL_KINDS = ("real", "social media")                 # conf's "real" side
    AI_SIDES = {   # conf's "AI" side; the first is the default
        "every generated set (Z-Image, VAE, half size)": lambda r: r["kind"] in ("fingerprint test", "other generated"),
        "Z-Image only (held-out blacks, colours, early tests, scenes)": lambda r: r["group"] in (
            "Z black, held out", "Z colours", "Z early flat tests", "Z scenes"),
    }
    CV_FOLDS, CV_SEED = 5, 0
    OUTCOMES = [(0.90, "Highly likely AI"), (0.60, "Likely AI"), (0.40, "Unsure"), (0.10, "Unlikely AI"),
                (0.00, "Real")]           # conf at or above each cut -> that outcome (first match)


@app.cell
def intro():
    mo.md(r"""
    # Results per picture

    Browses a results file written by `experiments/score_all.py` (controls.py sections 7 and 9: every test
    picture scored against the held-out-free Z-Image black fingerprint). Nothing is recomputed here except the
    thumbnails.

    - **score**: best aligned correlation of the picture's 16-px fold with any Z-Image black signature
    - **strength**: slope of the fold on the Z 1024 signature (1 = as strong as an average clean 1024² black)
    - **over threshold** (▲): above the synthetic-noise thresholds of that run (shown under the file picker)
    - **Conf**: how likely the picture is AI, from a logistic regression on **strength + score**, fitted with an
      **AI side** (picked below: every generated set, or Z-Image only) and the **real side** = every real and
      social-media picture, the two sides weighted equally (so it reads as if AI and real were equally common).
      Each labelled picture's conf comes from **5-fold cross-validation** (a model that never saw it); pictures on
      neither side get it from a model fitted on all labelled pictures
    - **Outcome**: conf in words: ≥ 0.90 highly likely AI, ≥ 0.60 likely AI, ≥ 0.40 unsure, ≥ 0.10 unlikely AI,
      below that real
    - **view**: the picture as it is, or its high-pass noise stretched to ±4 std (what the fold works on)
    """)
    return


@app.cell
def pick_file():
    _files = sorted(RESULTS.glob("*.json"), reverse=True)
    file_pick = mo.ui.dropdown({f.name: str(f) for f in _files}, value=_files[0].name if _files else None,
                               label="results file")
    file_pick
    return (file_pick,)


@app.cell
def load_results(file_pick):
    mo.stop(file_pick.value is None, mo.md(f"No results files in `{RESULTS}` yet."))
    _d = json.loads(Path(file_pick.value).read_text(encoding="utf-8"))
    meta, rows = _d["meta"], _d["rows"]
    records = {r.id: r for r in load_manifest(DATA)}
    mo.md(f"**{len(rows)} pictures** · made {meta['created']} · reference {meta['reference']} · "
          f"{meta['residual']} residual σ {meta['sigma']}, {meta['period']}-px fold · thresholds: correlation "
          f"**{meta['correlation_threshold']:.4f}**, strength **{meta['strength_threshold']:.4f}** "
          f"({meta['threshold_percentile']}th percentile of synthetic noise)"
          + (f"  \n**scale scan:** {meta['scan'].get('calibration', '')}; {meta['scan'].get('note', '')}"
             if "scan" in meta else ""))
    return meta, records, rows


@app.cell
def conf_side():
    side_pick = mo.ui.radio(list(AI_SIDES), value=next(iter(AI_SIDES)), label="Conf: AI side of the fit")
    side_pick
    return (side_pick,)


@app.cell
def conf_compute(rows, side_pick):
    if all("conf" in r for r in rows):        # a file that carries its own conf (rescore_scan.py combined)
        conf = {r["id"]: float(r["conf"]) for r in rows}
        conf_model = None
        conf_ai_ids = {r["id"] for r in rows if r["kind"] in ("fingerprint test", "other generated")}
    else:
        _is_ai = AI_SIDES[side_pick.value]
        _X = np.array([[r["strength"], r["score"]] for r in rows])
        _ai = np.array([_is_ai(r) for r in rows])
        _lab = _ai | np.array([r["kind"] in REAL_KINDS for r in rows])
        _p = np.empty(len(rows))
        _p[_lab], _ = fl.cv_logistic(_X[_lab], _ai[_lab].astype(float), k=CV_FOLDS, seed=CV_SEED)
        conf_model = fl.fit_logistic(_X[_lab], _ai[_lab].astype(float))
        if (~_lab).any():
            _p[~_lab] = fl.predict_logistic(conf_model, _X[~_lab])
        conf = {r["id"]: float(v) for r, v in zip(rows, _p)}
        conf_ai_ids = {r["id"] for r, a in zip(rows, _ai) if a}
    return conf, conf_ai_ids, conf_model


@app.cell
def conf_fit(conf, conf_ai_ids, conf_model, meta, rows, side_pick):
    _ai = np.array([conf[r["id"]] for r in rows if r["id"] in conf_ai_ids])
    _re = np.array([conf[r["id"]] for r in rows if r["kind"] in REAL_KINDS])
    _auc = float(((_ai[:, None] > _re[None, :]).sum() + 0.5 * (_ai[:, None] == _re[None, :]).sum()) / (len(_ai) * len(_re)))
    if conf_model is None:
        _txt = (f"conf comes from the results file ({meta['conf']['model']}; features "
                f"{', '.join(meta['conf']['features'])}; weights {meta['conf']['weights']})" if "conf" in meta
                else "conf comes from the results file")
        _side_txt = meta.get("conf", {}).get("ai_side", "the file's AI side")
    else:
        _b = conf_model["beta"]
        _txt = (f"Weights on standardised inputs (fit on all labelled pictures): intercept {_b[0]:.2f}, "
                f"strength {_b[1]:.2f}, score {_b[2]:.2f} · inputs standardised with mean "
                f"({conf_model['mu'][0]:.3f}, {conf_model['mu'][1]:.3f}) and sd "
                f"({conf_model['sd'][0]:.3f}, {conf_model['sd'][1]:.3f})")
        _side_txt = side_pick.value
    mo.accordion({f"conf fit · cross-validated AUC {_auc:.3f}": mo.vstack([
        mo.ui.table([
            {"side": f"AI: {_side_txt}", "pictures": len(_ai), "conf > 0.5": int((_ai > 0.5).sum()),
             "share": f"{100 * np.mean(_ai > 0.5):.0f}%", "median conf": round(float(np.median(_ai)), 3)},
            {"side": "real + social media", "pictures": len(_re), "conf > 0.5": int((_re > 0.5).sum()),
             "share": f"{100 * np.mean(_re > 0.5):.0f}%", "median conf": round(float(np.median(_re)), 3)},
        ], selection=None),
        mo.md(_txt),
    ])})
    return


@app.cell
def summary(meta, rows):
    _by = {}
    for _r in rows:
        _by.setdefault((_r["kind"], _r["group"]), []).append(_r)
    _t = []
    for (_k, _g), _rs in _by.items():
        _b = np.array([r["strength"] for r in _rs])
        _c = np.array([r["score"] for r in _rs])
        _t.append({"kind": _k, "group": _g, "pictures": len(_rs),
                   "strength median": round(float(np.median(_b)), 4), "strength max": round(float(_b.max()), 4),
                   "over strength thr.": f"{100 * np.mean(_b > meta['strength_threshold']):.0f}%",
                   "score median": round(float(np.median(_c)), 3),
                   "over score thr.": f"{100 * np.mean(_c > meta['correlation_threshold']):.0f}%"})
    mo.accordion({"summary per group": mo.ui.table(_t, selection=None, page_size=30)})
    return


@app.cell
def filters(rows):
    _kinds = list(dict.fromkeys(r["kind"] for r in rows))
    _groups = list(dict.fromkeys(r["group"] for r in rows))
    _default = [g for g in _groups if not g.startswith(("Z ", "VAE"))] or _groups[:1]
    group_pick = mo.ui.multiselect(_groups, value=_default, label="groups")
    sort_pick = mo.ui.dropdown(["conf, high first", "conf, low first", "strength, high first", "strength, low first",
                                "score, high first", "score, low first", "group, then strength"],
                               value="strength, high first", label="sort")
    view_pick = mo.ui.radio(["picture", "noise (stretched)"], value="picture", inline=True, label="view")
    per_page = mo.ui.slider(6, 60, value=24, step=6, label="per page", show_value=True)
    mo.vstack([mo.hstack([group_pick], justify="start"),
               mo.hstack([sort_pick, view_pick, per_page], justify="start"),
               mo.md(f"kinds in this file: {', '.join(_kinds)}")])
    return group_pick, per_page, sort_pick, view_pick


@app.cell
def selection(conf, group_pick, per_page, rows, sort_pick):
    _sel = [r for r in rows if r["group"] in group_pick.value]
    _key = {"conf, high first": lambda r: -conf[r["id"]], "conf, low first": lambda r: conf[r["id"]],
            "strength, high first": lambda r: -r["strength"], "strength, low first": lambda r: r["strength"],
            "score, high first": lambda r: -r["score"], "score, low first": lambda r: r["score"],
            "group, then strength": lambda r: (r["group"], -r["strength"])}[sort_pick.value]
    shown = sorted(_sel, key=_key)
    pages = max(1, -(-len(shown) // per_page.value))
    page_pick = mo.ui.number(start=1, stop=pages, value=1, label=f"page (of {pages}, {len(shown)} pictures)")
    page_pick
    return page_pick, shown


@app.cell
def thumbs(records):
    @functools.lru_cache(maxsize=512)
    def thumbnail(rid, view, size):
        """A size-px PNG of one stored picture, or of its high-pass noise stretched to ±4 std."""
        rec = records[rid]
        im = Image.open(resolve(DATA, rec)).convert("RGB")
        if view != "picture":
            px = np.asarray(im)
            if px.shape[0] * px.shape[1] > 3_000_000:      # big phone photos: noise view from the centre crop
                h, w = px.shape[:2]
                px = px[h // 2 - 768:h // 2 + 768, w // 2 - 768:w // 2 + 768]
            im = Image.fromarray(fl.stretched(fl.residuals(px[None], "highpass", None, 3.0)[0], 4)[0])
        im.thumbnail((size, size))
        b = io.BytesIO()
        im.save(b, "PNG")
        return b.getvalue()

    def _fit(im, mode):
        """What the preview shows: the whole image (downsampled if its long side is over PREVIEW_MAX) or, for
        actual pixels, a full-resolution PREVIEW_MAX square from the centre. Returns (image, note)."""
        w, h = im.size
        if mode == "whole picture":
            if max(w, h) <= PREVIEW_MAX:
                return im, ""
            f = PREVIEW_MAX / max(w, h)
            return im.resize((round(w * f), round(h * f)), Image.Resampling.LANCZOS), f"shown at {f:.0%} of {w}×{h}"
        if max(w, h) <= PREVIEW_MAX:
            return im, ""
        cw, ch = min(w, PREVIEW_MAX), min(h, PREVIEW_MAX)
        x, y = (w - cw) // 2, (h - ch) // 2
        return im.crop((x, y, x + cw, y + ch)), f"centre {cw}×{ch} crop of {w}×{h}, actual pixels"

    def _png(im):
        b = io.BytesIO()
        im.save(b, "PNG")
        return b.getvalue()

    @functools.lru_cache(maxsize=8)
    def full_picture(rid, mode):
        """The picture for the preview: its own file bytes when they fit untouched and a browser can show them
        (JPEG, PNG, WebP); otherwise (HEIC, or bigger than PREVIEW_MAX) a lossless PNG of what _fit shows.
        Returns (bytes, mime type, note)."""
        path = resolve(DATA, records[rid])
        im = Image.open(path)
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        if path.suffix.lower() in mime and max(im.size) <= PREVIEW_MAX:
            return path.read_bytes(), mime[path.suffix.lower()], ""
        shown, note = _fit(im.convert("RGB"), mode)
        return _png(shown), "image/png", note

    @functools.lru_cache(maxsize=8)
    def full_noise(rid, mode):
        """The picture's high-pass noise, stretched to ±4 std, for the same view as full_picture (computed at full
        resolution, then fitted the same way). Returns (bytes, note)."""
        im = Image.open(resolve(DATA, records[rid])).convert("RGB")
        if mode != "whole picture":                 # crop first: the noise of the shown pixels, at full resolution
            im, _ = _fit(im, mode)
        px = np.asarray(im)
        noise = Image.fromarray(fl.stretched(fl.residuals(px[None], "highpass", None, 3.0)[0], 4)[0])
        shown, note = _fit(noise, mode) if mode == "whole picture" else (noise, "")
        return _png(shown), note

    return full_noise, full_picture, thumbnail


@app.cell
def preview_state():
    get_preview, set_preview = mo.state(None)
    return get_preview, set_preview


@app.cell
def preview_controls(set_preview):
    fit_pick = mo.ui.radio(["whole picture", "actual pixels"], value="whole picture", inline=True, label="preview size")
    close_preview = mo.ui.button(label="✕ close preview", on_click=lambda _: set_preview(None))
    return close_preview, fit_pick


@app.cell
def preview(close_preview, conf, fit_pick, full_noise, full_picture, get_preview, records, rows):
    import base64 as _b64
    import html as _html

    _rid = get_preview()
    mo.stop(_rid is None)
    _r = next(r for r in rows if r["id"] == _rid)
    _rec = records[_rid]
    with mo.status.spinner("loading the picture and its noise"):
        _pic, _mime, _note = full_picture(_rid, fit_pick.value)
        _noise, _ = full_noise(_rid, fit_pick.value)
    _w, _h = Image.open(resolve(DATA, _rec)).size
    if fit_pick.value == "whole picture":       # scaled down to fit its half of the panel, never cropped or zoomed
        _box = "height:60vh;display:flex;align-items:center;justify-content:center"
        _img = "max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain"
    else:                                         # 1 image pixel = 1 screen pixel, scroll to look around
        _box, _img = "height:60vh;overflow:auto", ""

    def _panel(title, data, mime):
        src = f"data:{mime};base64," + _b64.b64encode(data).decode()
        return (f'<div style="flex:1 1 0;min-width:0"><div style="font-weight:600;margin-bottom:4px">{title}</div>'
                f'<div style="{_box};border:1px solid rgba(255,255,255,.25);background:rgba(0,0,0,.25)">'
                f'<img src="{src}" style="{_img};display:block"></div></div>')

    _row = {**_r, "conf": round(conf[_rid], 3), "outcome": next(n for c, n in OUTCOMES if conf[_rid] >= c),
            "set": _rec.set, "source": _rec.source, "file": _rec.file}
    _cells = "".join(f'<tr><td style="padding:2px 12px 2px 0;opacity:.7">{_html.escape(str(k))}</td>'
                     f'<td style="padding:2px 0">{_html.escape(str(v))}</td></tr>' for k, v in _row.items())
    # a full-screen layer over the page: dark backdrop, the preview on a panel in the middle
    mo.Html(
        f'<div style="position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.78);overflow:auto;padding:4vh 24px;box-sizing:border-box">'
        f'<div style="max-width:1100px;max-height:92vh;overflow:auto;margin:0 auto;background:var(--background, #1e1e1e);'
        f'color:var(--foreground, #eee);border-radius:10px;padding:12px 16px;box-shadow:0 10px 40px rgba(0,0,0,.5)">'
        f'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:10px">'
        f'<div style="flex:1"><div style="font-size:16px;font-weight:700">{_html.escape(TITLE.get(_r["kind"], _r["kind"]))} · {_rid} · {_w}×{_h} px</div>'
        f'<div style="font-size:12px;opacity:.7">{_html.escape(_note) or "full definition"}</div></div>'
        f'{fit_pick}{close_preview}</div>'
        f'<div style="display:flex;gap:12px;align-items:flex-start">'
        f'{_panel("picture", _pic, _mime)}'
        f'{_panel("noise, stretched ±4 std", _noise, "image/png")}</div>'
        f'<table style="margin-top:10px;font-size:12px;border-collapse:collapse">{_cells}</table>'
        f'</div></div>')
    return


@app.cell
def grid(conf, meta, page_pick, per_page, records, set_preview, shown, thumbnail, view_pick):
    import base64
    import html

    def _outcome(c):
        return next(name for cut, name in OUTCOMES if c >= cut)

    _start = (page_pick.value - 1) * per_page.value
    _page = shown[_start:_start + per_page.value]

    def _thumb_label(r):
        if r["id"] not in records:
            return "not in this store"
        src = "data:image/png;base64," + base64.b64encode(thumbnail(r["id"], view_pick.value, THUMB)).decode()
        return (f'<img src="{src}" title="click for a full-size preview" '
                f'style="max-width:{THUMB}px;max-height:{THUMB}px;object-fit:contain;cursor:zoom-in">')

    # one button per card: its label is the thumbnail, a click puts that picture in the preview above
    card_buttons = mo.ui.array([
        mo.ui.button(label=_thumb_label(r), kind="neutral", on_click=lambda _, rid=r["id"]: set_preview(rid))
        for r in _page])

    def _card(r, button):
        over_b = "▲" if r["strength"] > meta["strength_threshold"] else "·"
        over_c = "▲" if r["score"] > meta["correlation_threshold"] else "·"
        line = 'style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis"'
        small = 'style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:11px;opacity:.75"'
        return (f'<div style="width:{CARD_W}px;height:{CARD_H + 16}px;box-sizing:border-box;padding:8px;'
                f'border:1px solid rgba(128,128,128,.35);border-radius:8px;overflow:hidden;font-size:13px">'
                f'<div style="width:{THUMB}px;height:{THUMB + 16}px;display:flex;align-items:center;justify-content:center">{button}</div>'
                f'<div style="margin-top:6px;font-weight:600" {line}>{html.escape(TITLE.get(r["kind"], r["kind"]))}:</div>'
                f'<div {line}>Conf: <b>{conf[r["id"]]:.2f}</b></div>'
                f'<div {line}>strength <b>{r["strength"]:.4f}</b> {over_b} · score {r["score"]:.3f} {over_c}</div>'
                f'<div {line}>Outcome: <b>{_outcome(conf[r["id"]])}</b></div>'
                f'<div style="height:8px"></div>'
                f'<div {small}>{r["size"]} · {r["id"]}'
                + (f' · ×{r["scale"]:.3f} (from ~{r["original width guess"]} px)' if "scale" in r else "")
                + '</div></div>')

    _cards = "".join(_card(r, b) for r, b in zip(_page, card_buttons))
    mo.vstack([mo.md("▲ = over that run's threshold · **Conf** = how likely AI (cross-validated logistic regression) · "
                     "**click a picture** for a full-size preview on top of the page (✕ close to go back)"),
               mo.Html(f'<div style="display:flex;flex-wrap:wrap;gap:12px">{_cards}</div>')])
    return (card_buttons,)


if __name__ == "__main__":
    app.run()
