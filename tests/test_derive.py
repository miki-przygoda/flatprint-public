import io

from PIL import Image

from flatprint.derive import derive_set
from flatprint.store import Shard, load_manifest, merge


def add_png(shard, id, size):
    b = io.BytesIO()
    Image.new("RGB", size, (50, 50, 50)).save(b, "PNG")
    return shard.add({"id": id, "set": "content", "model": "m", "backend": "diffusers"}, b.getvalue(), "png")


def test_derive_links_parents_and_skips_small(data_root):
    src = Shard(data_root, "mac")
    add_png(src, "a" * 16, (64, 64))
    add_png(src, "b" * 16, (16, 16))                     # too small for crop:32
    merge(data_root)
    out = Shard(data_root, "host-b")
    s = derive_set(load_manifest(data_root), data_root, out, "crop:32,jpeg:90", "content-crop32-q90")
    assert (s.done, s.failed) == (1, 1)
    rec = out.records()[0]
    assert rec.parent_id == "a" * 16 and rec.ops == "crop:32,jpeg:90" and rec.set == "content-crop32-q90"
    assert rec.file.endswith(".jpg") and (rec.width, rec.height) == (32, 32)
    assert "b" * 16 in (out.dir / "errors.jsonl").read_text()


def test_derive_is_resumable(data_root):
    add_png(Shard(data_root, "mac"), "a" * 16, (64, 64))
    merge(data_root)
    out = Shard(data_root, "host-b")
    derive_set(load_manifest(data_root), data_root, out, "jpeg:50", "q50")
    s = derive_set(load_manifest(data_root), data_root, Shard(data_root, "host-b"), "jpeg:50", "q50")
    assert (s.done, s.skipped) == (0, 1)
