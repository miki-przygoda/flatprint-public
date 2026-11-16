from flatprint.importer import import_dir
from flatprint.store import Shard


def write_imgs(d, tiny_rgb, names):
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        tiny_rgb(colour=(i, i, i)).save(d / n)


def test_import_copies_bytes_and_is_idempotent(tmp_path, data_root, tiny_rgb):
    src = tmp_path / "coco"
    write_imgs(src, tiny_rgb, ["a.jpg", "b.png"])
    (src / "notes.txt").write_text("not an image")
    shard = Shard(data_root, "mac")
    s1 = import_dir(src, shard, set="real", source="COCO val2017", licence="CC BY 4.0")
    s2 = import_dir(src, shard, set="real", source="COCO val2017", licence="CC BY 4.0")
    assert (s1.done, s2.done, s2.skipped) == (2, 0, 2)
    recs = shard.records()
    assert {r.extra["original_name"] for r in recs} == {"a.jpg", "b.png"}
    assert {r.file.rsplit(".", 1)[1] for r in recs} == {"jpg", "png"}
    for r in recs:
        orig = src / r.extra["original_name"]
        assert (shard.dir / r.file).read_bytes() == orig.read_bytes()
        assert r.source == "COCO val2017" and r.licence == "CC BY 4.0" and r.backend == "import"


def test_link_parents_by_filename_prefix(tmp_path, data_root, tiny_rgb):
    parent = "0123456789abcdef"
    src = tmp_path / "social"
    write_imgs(src, tiny_rgb, [f"{parent}-instagram.jpg", "unrelated.jpg"])
    shard = Shard(data_root, "mac")
    import_dir(src, shard, set="social-instagram", source="instagram round trip", licence=None,
               link_parents=True, known_ids={parent})
    by_name = {r.extra["original_name"]: r for r in shard.records()}
    assert by_name[f"{parent}-instagram.jpg"].parent_id == parent
    assert by_name["unrelated.jpg"].parent_id is None


def test_heic_imports_unchanged_and_opens_with_pil(tmp_path, data_root, tiny_rgb):
    import numpy as np
    import pillow_heif
    from PIL import Image

    import flatprint  # noqa: F401  (registers the HEIF opener)
    from flatprint.backends.base import sniff_ext
    src = tmp_path / "phone"
    src.mkdir()
    pillow_heif.from_pillow(tiny_rgb(64, 48, (40, 80, 120))).save(src / "IMG_0001.HEIC", quality=-1)
    shard = Shard(data_root, "mac")
    assert import_dir(src, shard, set="real-phone", source="phone", licence=None).done == 1
    rec = shard.records()[0]
    stored = shard.dir / rec.file
    assert rec.file.endswith(".heic") and (rec.width, rec.height) == (64, 48)
    assert stored.read_bytes() == (src / "IMG_0001.HEIC").read_bytes()
    assert sniff_ext(stored.read_bytes()) == "heic"
    px = np.asarray(Image.open(stored).convert("RGB")).astype(int)
    assert px.shape == (48, 64, 3) and np.abs(px.mean(axis=(0, 1)) - (40, 80, 120)).max() < 3
