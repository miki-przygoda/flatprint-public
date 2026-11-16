"""Metadata-rule baselines: what a detector that only reads file metadata gets on the report's test groups.

Rule G ("generator tag"): AI if the file carries generator metadata (PNG text chunks, XMP, C2PA).
Rule E ("no camera EXIF"): AI if the file has no EXIF block (HEIC: an Exif item anywhere in the file).
Both are binary, so AUC = (TPR + TNR) / 2 against the same 36 social-media real pictures as Tables 2-4.

    uv run python experiments/metadata_baseline.py      # -> logs/results/metadata-baseline.csv + printed table
"""
import os
from pathlib import Path

from flatprint.dataset import read_ids
from flatprint.metadata import inspect_bytes
from flatprint.store import load_manifest, resolve

ROOT = Path(__file__).parent.parent
DATA = Path(os.environ.get("FLATPRINT_DATA", ROOT / "data"))
OUT = ROOT / "logs" / "results" / "metadata-baseline.csv"
NEGATIVE = ["web-black-pinterest", "real-social", "real-social-hard"]       # the 36 social real
GROUPS = {  # report row -> sets
    "Z black (all clean)": ["flat-black-sizes"],
    "Z scenes (showcase)": ["showcase", "adhoc"], "Z scenes (COCO)": ["z-scenes"],
    "Scenes, JPEG 85": ["showcase-q85"], "Scenes, 736 w": ["showcase-w736"],
    "Scenes, 736 w + JPEG 85": ["showcase-w736-q85"], "Scenes, 564 w + JPEG 85": ["showcase-w564-q85"],
    "FLUX black": ["flux-black"], "FLUX scenes": ["flux-scenes"],
    "[real] phone originals": ["phone-blank", "real-phone", "real-phone-dim"],
    "[real] phone, resized copies": ["real-phone-w736-q85", "real-phone-w564-q85",
                                     "real-phone-dim-w736-q85", "real-phone-dim-w564-q85"],
}


def flags(path: Path) -> tuple[bool, bool]:
    data = path.read_bytes()
    m = inspect_bytes(data)
    generator = bool(m["text_keys"]) or m["xmp"] or m["c2pa"]
    exif = m["exif"] or (m["format"] == "heic" and b"Exif\x00\x00" in data)
    return generator, not exif


def main():
    skip = set().union(*(read_ids(p) for p in DATA.glob("*-flagged/flagged.md")))
    man = load_manifest(DATA)

    def rows(sets):
        return [flags(resolve(DATA, r)) for r in man if r.set in sets and r.id not in skip]

    neg = rows(NEGATIVE)
    neg_rate = [sum(f[k] for f in neg) / len(neg) for k in (0, 1)]       # false-alarm rate per rule
    lines = ["group,n,rule G flagged,rule G AUC,rule E flagged,rule E AUC"]
    print(f"negatives: {len(neg)} social real; rule G flags {neg_rate[0]:.3f}, rule E flags {neg_rate[1]:.3f}")
    print(f"{'group':32s} {'n':>4s}  G flag  G AUC   E flag  E AUC")
    for name, sets in GROUPS.items():
        pos = rows(sets)
        rate = [sum(f[k] for f in pos) / len(pos) for k in (0, 1)]
        auc = [(rate[k] + 1 - neg_rate[k]) / 2 for k in (0, 1)]
        lines.append(f"{name},{len(pos)},{rate[0]:.3f},{auc[0]:.3f},{rate[1]:.3f},{auc[1]:.3f}")
        print(f"{name:32s} {len(pos):4d}  {rate[0]:.3f}  {auc[0]:.3f}   {rate[1]:.3f}  {auc[1]:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
