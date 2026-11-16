"""Import existing image files (bytes unchanged) into a shard with provenance."""

import hashlib
import re
from pathlib import Path

from PIL import Image

from flatprint.backends.base import sniff_ext
from flatprint.runner import RunSummary
from flatprint.store import Shard

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
ID_PREFIX = re.compile(r"^([0-9a-f]{16})")


def import_dir(src: Path, shard: Shard, set: str, source: str, licence: str | None,
               link_parents: bool = False, known_ids: set[str] | None = None) -> RunSummary:
    s = RunSummary()
    for path in sorted(p for p in Path(src).rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        data = path.read_bytes()
        id = hashlib.sha256(data).hexdigest()[:16]
        if shard.has(id):
            s.skipped += 1
            continue
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception as e:  # noqa: BLE001
            shard.log_error(id, f"{path.name}: unreadable image: {e}")
            s.failed += 1
            continue
        parent = None
        if link_parents:
            m = ID_PREFIX.match(path.stem)
            if m and (known_ids is None or m.group(1) in known_ids):
                parent = m.group(1)
        ext = sniff_ext(data)
        if ext == "bin":
            ext = path.suffix.lstrip(".").lower()
        shard.add({"id": id, "set": set, "model": source, "backend": "import", "width": w, "height": h,
                   "source": source, "licence": licence, "parent_id": parent,
                   "extra": {"original_name": path.name}}, data, ext)
        s.done += 1
    return s
