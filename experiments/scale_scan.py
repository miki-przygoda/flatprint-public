"""Scale-aware scoring: each picture is scored at every guess of the original width it was shrunk from, the
best guess (highest strength) is kept, and that best is calibrated against chance for the number of guesses.
Runs controls.py's scoring first for its held-out-free signatures.

    SCAN_WORKERS=8 uv run python experiments/scale_scan.py OUT.json

Guesses: as it is, plus every usual generator width (WIDTHS: multiples of 64 from 512 to 2048) above the
picture's width. Calibration: a picture that tried k scales is compared with the best of k values drawn from the
pooled per-scale strengths of every real and social-media picture (fingerprint_lib.best_of_k_null), as a z-score.
OUT.json has results.py's format: "strength" is that z-score, "score" the correlation at the best guess, with
"strength best raw", "scale", "original width guess", "strength as is" and "score as is" beside them.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fingerprint_lib as fl  # noqa: E402

SCAN_KINDS = ("real", "social media", "unlabelled", "AI, other generator", "robustness test", "fingerprint test",
              "FLUX (same VAE)")
SCAN_GROUPS = ("Z scenes", "Z scenes (COCO)", "Z black, half size", "Z scenes (COCO), 736 wide",
               "Z scenes (COCO), 736 wide + JPEG 85", "Z scenes (COCO), 564 wide + JPEG 85")   # of the other generated groups; half size:
                                                                      # held-out parents only
MAX_WIDTH = 2048
WIDTHS = range(512, MAX_WIDTH + 1, 64)      # the usual output widths of SDXL, FLUX and Z-Image presets
REAL_KINDS = ("real", "social media")
Z_THRESHOLD = 2.326                          # one-sided 1 % of a normal: the viewer's strength threshold for z
_W = {}


def _init(sigs, ref, mode, sigma, period):
    _W.update(sigs=sigs, ref=ref, mode=mode, sigma=sigma, period=period)


def _scan(path):
    import flatprint  # noqa: F401  (HEIC opener)
    from PIL import Image
    px = np.asarray(Image.open(path).convert("RGB"))
    return fl.scale_scan(px, fl.candidate_scales(px.shape[1], widths=WIDTHS), _W["sigs"], _W["ref"],
                         _W["mode"], _W["sigma"], _W["period"], MAX_WIDTH)


def main(out: Path) -> None:
    sys.argv = [sys.argv[0], "--all"]
    import controls
    from flatprint.store import load_manifest, resolve

    _, d = controls.app.run()
    records = {r.id: r for r in load_manifest(controls.DATA)}
    held = {r["id"] for r in d["verify_rows"] if r["group"] == "Z black, held out"}
    rows = [r for r in d["verify_rows"]
            if (r["kind"] in SCAN_KINDS or r["group"] in SCAN_GROUPS) and r["id"] in records
            and (r["group"] != "Z black, half size" or records[r["id"]].parent_id in held)]
    paths = [str(resolve(controls.DATA, records[r["id"]])) for r in rows]
    sigs = {k: np.asarray(v) for k, v in d["verify_sigs"].items()}
    args = (sigs, sigs[controls.REF], d["mode_pick"].value, d["sigma_pick"].value, d["period_pick"].value)
    workers = int(os.environ.get("SCAN_WORKERS", "4"))
    print(f"scanning {len(rows)} pictures with {workers} workers", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=args) as pool:
        scans = list(pool.map(_scan, paths, chunksize=2))
    pool = [x["strength"] for r, scan in zip(rows, scans) if r["kind"] in REAL_KINDS for x in scan]
    nulls = {k: fl.best_of_k_null(pool, k, seed=k) for k in {len(scan) for scan in scans}}
    print(f"null pool: {len(pool)} per-scale strengths from real pictures; "
          + ", ".join(f"k={k}: {m:.4f}±{sd:.4f}" for k, (m, sd) in sorted(nulls.items())[:6]), flush=True)
    out_rows = []
    for r, scan in zip(rows, scans):
        best = max(scan, key=lambda x: x["strength"])
        as_is = next(x for x in scan if x["scale"] == 1.0)
        width = int(r["size"].split("x")[0])
        m, sd = nulls[len(scan)]
        out_rows.append({**r, "strength": round((best["strength"] - m) / sd, 3), "score": round(best["score"], 3),
                         "strength best raw": round(best["strength"], 4), "scale": round(best["scale"], 4),
                         "original width guess": round(width / best["scale"]), "scales tried": len(scan),
                         "strength as is": round(as_is["strength"], 4), "score as is": round(as_is["score"], 3),
                         "scan strengths": [round(x["strength"], 4) for x in scan]})
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "correlation_threshold": d["threshold"], "strength_threshold": Z_THRESHOLD,
            "reference": controls.REF, "residual": args[2], "sigma": args[3], "period": args[4],
            "holdout_per_group": d["holdout_pick"].value, "holdout_seed": d["holdout_seed"].value,
            "threshold_percentile": d["pct_pick"].value,
            "scan": {"widths": [WIDTHS.start, WIDTHS.stop - 1, WIDTHS.step], "best_by": "strength",
                     "calibration": "z of the best strength vs the best of k draws from real pictures' per-scale strengths",
                     "note": "strength is that z (threshold one-sided 1 %); score is the raw correlation at the best scale "
                             "and its threshold the unscanned synthetic one"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": out_rows}), encoding="utf-8")
    print(f"{len(out_rows)} rows -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
