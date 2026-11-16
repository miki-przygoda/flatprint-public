"""Re-score a scale scan (scale_scan.py output) from its saved per-scale strengths, two ways, without rescanning:

  size-matched: z of each picture's best strength against the best-of-scan strengths of real pictures that tried
                a similar number of scales (fingerprint_lib.matched_z, within TOLERANCE, leaving itself out)
  sharpness:    how far its best scale stands out from its own other scales (fingerprint_lib.peak_sharpness);
                pictures with fewer than 4 scales (wider than ~1800 px) get 0, "no evidence of a resize"

    uv run python experiments/rescore_scan.py SCAN.json

  combined:     a logistic regression (balanced classes) on strength as is, score, size-matched z and sharpness,
                AI side = the scanned Z-Image groups (held-out blacks, scenes, half size), real side = every real
                and social-media picture; 5-fold cross-validation grouped by lineage (an original and all its
                derived copies share one fold), so every picture's conf, robustness copies included, comes from a
                model that never saw any picture of its lineage

writes SCAN-sizematched.json, SCAN-sharpness.json (results.py's format, "strength" = the new value),
SCAN-combined.json and SCAN-combined-nosharp.json (the same classifier without sharpness; each with every picture's
"conf", which results.py then uses) and prints AUCs against the
size-matched social-media real pictures and against every real picture.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fingerprint_lib as fl  # noqa: E402

REAL_KINDS = ("real", "social media")
AI_KINDS = ("fingerprint test", "other generated")   # every scanned Z-Image group, the COCO scenes included
FEATURES = ("strength as is", "score", "size-matched z", "sharpness")
TOLERANCE = 3
Z_THRESHOLD = 2.326
GROUPS = ["Z scenes", "robust: scenes, JPEG 85", "robust: scenes, 736 wide", "robust: scenes, 736 wide + JPEG 85",
          "robust: scenes, 564 wide + JPEG 85", "Z black, held out", "robust: held-out blacks, JPEG 85",
          "robust: held-out blacks, 736 wide", "robust: held-out blacks, 736 wide + JPEG 85",
          "robust: held-out blacks, 564 wide + JPEG 85", "Z black, half size", "social media, judged likely AI",
          "social media, maybe AI", "Z scenes (COCO)", "Z scenes (COCO), 736 wide", "Z scenes (COCO), 736 wide + JPEG 85",
          "Z scenes (COCO), 564 wide + JPEG 85", "FLUX black", "FLUX scenes"]


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main(scan_path: Path) -> None:
    d = json.loads(scan_path.read_text(encoding="utf-8"))
    rows, meta = d["rows"], d["meta"]
    refs = [(r["scales tried"], r["strength best raw"]) for r in rows if r["kind"] in REAL_KINDS]
    matched, sharp, missing = {}, {}, 0
    for r in rows:
        own = (r["scales tried"], r["strength best raw"]) if r["kind"] in REAL_KINDS else None
        z = fl.matched_z(r["strength best raw"], r["scales tried"], refs, TOLERANCE, leave_out=own)
        if np.isnan(z):
            missing += 1
            z = 0.0
        matched[r["id"]] = z
        s = fl.peak_sharpness(r["scan strengths"])
        sharp[r["id"]] = 0.0 if np.isnan(s) else float(min(s, 50.0))
    real_sharp = [sharp[r["id"]] for r in rows if r["kind"] in REAL_KINDS]
    sharp_threshold = float(np.percentile(real_sharp, 99))

    versions = {"as is": lambda r: r["strength as is"], "pooled z": lambda r: r["strength"],
                "size-matched z": lambda r: matched[r["id"]], "sharpness": lambda r: sharp[r["id"]]}
    soc = [r for r in rows if r["kind"] == "social media"]
    allr = [r for r in rows if r["kind"] in REAL_KINDS]
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    print(f"{len(rows)} pictures; size-matched z had no reference group for {missing} (set to 0)")
    for title, pool in ((f"AUC vs social-media real ({len(soc)}, size-matched)", soc), (f"AUC vs all real ({len(allr)})", allr)):
        print(title)
        print(f"  {'group':44s} " + " ".join(f"{k:>15s}" for k in versions))
        for g in GROUPS:
            if by[g]:
                print(f"  {g:44s} " + " ".join(f"{_auc([f(r) for r in by[g]], [f(r) for r in pool]):15.3f}"
                                               for f in versions.values()))
    for name, vals, thr in (("size-matched z", matched, Z_THRESHOLD), ("sharpness", sharp, sharp_threshold)):
        real_over = sum(vals[r["id"]] > thr for r in allr)
        print(f"{name}: threshold {thr:.3f}; real over it {real_over}/{len(allr)}; "
              + "; ".join(f"{g.removeprefix('robust: ')} {100 * np.mean([vals[r['id']] > thr for r in by[g]]):.0f}%"
                          for g in GROUPS[:5] if by[g]))

    for suffix, vals, thr, how in (
            ("sizematched", matched, Z_THRESHOLD,
             f"z of the best strength vs real pictures' best-of-scan within ±{TOLERANCE} scales (leave-one-out)"),
            ("sharpness", sharp, sharp_threshold,
             "peak sharpness: (best - median of the other scales) / (1.4826 x MAD); < 4 scales -> 0")):
        out_rows = [{**r, "strength": round(vals[r["id"]], 3), "strength (pooled z)": r["strength"]} for r in rows]
        out_meta = {**meta, "strength_threshold": thr,
                    "scan": {**meta.get("scan", {}), "calibration": how,
                             "note": "strength is this re-scored value; threshold "
                                     + ("one-sided 1 %" if suffix == "sizematched" else "the real pictures' 99th percentile")}}
        out = scan_path.with_name(f"{scan_path.stem}-{suffix}.json")
        out.write_text(json.dumps({"meta": out_meta, "rows": out_rows}), encoding="utf-8")
        print(f"-> {out}")

    # combined: one cross-validated classifier, with and without sharpness
    is_ai = np.array([r["kind"] in AI_KINDS for r in rows])
    lab = is_ai | np.array([r["kind"] in REAL_KINDS for r in rows])
    from flatprint.store import load_manifest
    parent = {r.id: r.parent_id for r in load_manifest(os.environ["FLATPRINT_DATA"])}
    lineage = np.array([parent.get(r["id"]) or r["id"] for r in rows])     # derived copy -> its original
    value = {"strength as is": lambda r: r["strength as is"], "score": lambda r: r["score"],
             "size-matched z": lambda r: matched[r["id"]], "sharpness": lambda r: sharp[r["id"]]}
    for suffix, feats in (("combined", FEATURES), ("combined-nosharp", tuple(f for f in FEATURES if f != "sharpness"))):
        X = np.array([[value[f](r) for f in feats] for r in rows])
        conf, _ = fl.grouped_cv_logistic(X, is_ai.astype(float), lab, lineage, k=5, seed=0)
        model = fl.fit_logistic(X[lab], is_ai[lab].astype(float))          # for the weights only
        cid = {r["id"]: float(c) for r, c in zip(rows, conf)}
        weights = dict(zip(("intercept",) + feats, [round(float(b), 3) for b in model["beta"]]))
        print(f"\n{suffix}: weights on standardised inputs {weights}")
        for title, pool in ((f"conf AUC vs social-media real ({len(soc)})", soc), (f"conf AUC vs all real ({len(allr)})", allr)):
            print(title)
            for g in GROUPS:
                if by[g]:
                    c = [cid[r["id"]] for r in by[g]]
                    print(f"  {g:44s} {_auc(c, [cid[r['id']] for r in pool]):6.3f}   conf > 0.5: {100 * np.mean(np.array(c) > 0.5):3.0f}%")
        fa = [r for r in allr if cid[r["id"]] > 0.5]
        print(f"real pictures with conf > 0.5: {len(fa)}/{len(allr)}  by group: "
              + ", ".join(f"{g} {sum(r['group'] == g for r in fa)}/{sum(r['group'] == g for r in allr)}"
                          for g in dict.fromkeys(r["group"] for r in allr)))
        out_rows = [{**r, "strength": round(matched[r["id"]], 3), "strength (pooled z)": r["strength"],
                     "size-matched z": round(matched[r["id"]], 3), "sharpness": round(sharp[r["id"]], 3),
                     "conf": round(cid[r["id"]], 4), "conf from": "lineage-grouped cross-validation",
                     "lineage": str(g)} for r, g in zip(rows, lineage)]
        out_meta = {**meta, "strength_threshold": Z_THRESHOLD,
                    "conf": {"model": "logistic regression, balanced classes, 5-fold CV grouped by lineage (original + copies)",
                             "features": list(feats), "weights": weights,
                             "ai_side": "scanned Z-Image groups (held-out blacks, scenes, half size, COCO scenes and their resized copies)",
                             "real_side": "every real and social-media picture (resized phone photos included)"},
                    "scan": {**meta.get("scan", {}), "calibration": "strength = size-matched z; conf = combined classifier",
                             "note": f"conf comes from this file ({', '.join(feats)})"}}
        out = scan_path.with_name(f"{scan_path.stem}-{suffix}.json")
        out.write_text(json.dumps({"meta": out_meta, "rows": out_rows}), encoding="utf-8")
        print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
