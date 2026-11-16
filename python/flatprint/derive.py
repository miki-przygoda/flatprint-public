"""Derived sets: apply an op chain to every record of a set, with provenance."""

import hashlib
import io
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from flatprint.backends.base import sniff_ext
from flatprint.ops import apply_ops, format_ops, parse_ops
from flatprint.runner import RunSummary
from flatprint.store import Record, Shard, resolve


def derive_set(records: list[Record], root: Path, shard: Shard, ops_spec: str, to_set: str,
               progress: bool = False) -> RunSummary:
    ops = parse_ops(ops_spec)
    ops_spec = format_ops(ops)
    s = RunSummary()
    for rec in tqdm(records, disable=not progress, desc=to_set):
        id = hashlib.sha256(f"{rec.id}|{to_set}|{ops_spec}".encode()).hexdigest()[:16]
        if shard.has(id):
            s.skipped += 1
            continue
        try:
            out = apply_ops(resolve(root, rec).read_bytes(), ops)
        except (ValueError, OSError) as e:
            shard.log_error(rec.id, f"derive '{ops_spec}': {e}")
            s.failed += 1
            continue
        w, h = Image.open(io.BytesIO(out)).size
        shard.add({"id": id, "set": to_set, "model": rec.model, "backend": "derive", "prompt": rec.prompt,
                   "seed": rec.seed, "width": w, "height": h, "source": rec.source, "licence": rec.licence,
                   "parent_id": rec.id, "ops": ops_spec, "extra": {"parent_set": rec.set}},
                  out, sniff_ext(out))
        s.done += 1
    return s
