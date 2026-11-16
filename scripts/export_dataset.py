"""Build the shareable dataset folder (for the Hugging Face dataset linked from the report).

    uv run python scripts/export_dataset.py OUT_DIR

- Every generated set (Z-Image-Turbo, FLUX.1-schnell, the Z-Image VAE round trips) and every laundered copy of them,
  as stored (hard links, so no extra disk), plus the own phone photos.
- Phone originals (JPEG / HEIC) are written as lossless PNG of the decoded pixels, which is what the analysis read,
  with no EXIF (some originals carry GPS). Their record keeps the original file's sha256, format and capture day
  (no time) under `extra`. The resized phone copies carry no EXIF and are shared as stored.
- Social-media downloads (Pinterest) are left out: third-party images.
- manifest.jsonl: the store's own format (one record per image, `file` relative to OUT_DIR, `sha256` of the shared
  file), and the flagged lists at the store's paths, so FLATPRINT_DATA=OUT_DIR runs the experiments unchanged.
"""
import hashlib
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import flatprint  # noqa: F401  (registers the HEIF opener)
from flatprint.store import load_manifest, resolve

DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
LEFT_OUT = {"web-black-pinterest", "real-social", "real-social-hard", "likely-ai-social", "ai-maybe-social"}
PHONE_ORIGINALS = {"phone-blank", "real-phone", "real-phone-dim"}
FLAGGED = ["flat-black-sizes-flagged", "flat-rgb-flagged"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def capture_day(img: Image.Image) -> str | None:
    exif = img.getexif()
    value = exif.get(0x0132) or exif.get_ifd(0x8769).get(0x9003)
    return value[:10].replace(":", "-") if value else None


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {}
    for r in load_manifest(DATA):
        if r.set in LEFT_OUT:
            continue
        src = resolve(DATA, r)
        rec = json.loads(json.dumps(dataclasses.asdict(r), default=str))
        if r.set in PHONE_ORIGINALS:
            dst = out / "images" / r.set / f"{r.id}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as img:
                day = capture_day(img)
                Image.fromarray(np.asarray(img.convert("RGB"))).save(dst, "PNG")   # pixels only, no EXIF
            rec["extra"] = {**(rec.get("extra") or {}), "original_sha256": r.sha256,
                            "original_format": src.suffix.lstrip(".").lower(), "capture_date": day}
        else:
            dst = out / "images" / r.set / f"{r.id}{src.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        rec["file"] = str(dst.relative_to(out))
        rec["sha256"] = sha256(dst)
        rows.append(rec)
        counts[r.set] = counts.get(r.set, 0) + 1
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    for name in FLAGGED:
        if (DATA / name / "flagged.md").exists():
            (out / name).mkdir(exist_ok=True)
            shutil.copy2(DATA / name / "flagged.md", out / name / "flagged.md")
    print(f"{len(rows)} images in {len(counts)} sets -> {out}")
    for s, n in sorted(counts.items()):
        print(f"  {n:5d}  {s}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
