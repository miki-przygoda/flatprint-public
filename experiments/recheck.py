"""Re-read saved scan results (scale_scan.py / rescore_scan.py output) and print AUCs for two subsets, without
rescanning:

  1. laundered held-out blacks with and without the 512 px parents (736 / 564 wide never enlarges, so a 512 parent's
     "resized" copy is the original, or the original + JPEG)
  2. "Z scenes" (showcase + adhoc) against showcase only, the 32 images the published detectors ran on (Table 4)

    uv run python experiments/recheck.py logs/results/SCAN.json [SCAN-sizematched.json ...]

The value scored per file: "conf" if the rows have it (-combined*), "strength" for -sizematched / -sharpness, and
"strength as is" otherwise (a plain scan). AUC against the social-media real pictures (36), as in Tables 2-4.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from flatprint.store import load_manifest  # noqa: E402

DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
BLACK_GROUPS = ["Z black, held out", "robust: held-out blacks, JPEG 85", "robust: held-out blacks, 736 wide",
                "robust: held-out blacks, 736 wide + JPEG 85", "robust: held-out blacks, 564 wide + JPEG 85"]
SCENE_GROUPS = ["Z scenes", "robust: scenes, JPEG 85", "robust: scenes, 736 wide",
                "robust: scenes, 736 wide + JPEG 85", "robust: scenes, 564 wide + JPEG 85"]


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def value_fn(path: Path, rows):
    if rows and "conf" in rows[0]:
        return "conf", lambda r: r["conf"]
    if path.stem.endswith(("-sizematched", "-sharpness")):
        return "strength (re-scored)", lambda r: r["strength"]
    return "strength as is", lambda r: r["strength as is"]


def main(paths: list[Path]) -> None:
    man = {r.id: r for r in load_manifest(DATA)}

    def width(i):                        # a row's original width: its parent's for a derived copy
        r = man.get(i)
        if r is None:
            return None
        p = man.get(r.parent_id) if r.parent_id else r
        return p.width if p else None

    def set_of(i):
        r = man.get(i)
        return r.set if r else None

    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        name, val = value_fn(path, rows)
        neg = [val(r) for r in rows if r["kind"] == "social media"]
        by = {}
        for r in rows:
            by.setdefault(r["group"], []).append(r)
        print(f"\n== {path.name}: {name}, AUC vs {len(neg)} social-media real")

        print("  1. held-out blacks                              all (n)        without 512 parents (n)   512 parents (n)")
        for g in BLACK_GROUPS:
            rs = by.get(g, [])
            unknown = sum(width(r["id"]) is None for r in rs)
            no512 = [r for r in rs if width(r["id"]) != 512]
            only512 = [r for r in rs if width(r["id"]) == 512]
            print(f"     {g:46s} {_auc([val(r) for r in rs], neg):.3f} ({len(rs):3d})   "
                  f"{_auc([val(r) for r in no512], neg):.3f} ({len(no512):3d})"
                  f"              {_auc([val(r) for r in only512], neg):.3f} ({len(only512):2d})"
                  + (f"   [{unknown} ids not in the manifest]" if unknown else ""))

        print("  2. scenes                                       all (n)        showcase only (n)")
        for g in SCENE_GROUPS:
            rs = by.get(g, [])
            sc = [r for r in rs if (set_of(r["id"]) or "").startswith("showcase")]
            print(f"     {g:46s} {_auc([val(r) for r in rs], neg):.3f} ({len(rs):3d})   "
                  f"{_auc([val(r) for r in sc], neg):.3f} ({len(sc):3d})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main([Path(p) for p in sys.argv[1:]])
