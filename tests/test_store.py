import json

from flatprint.store import Shard, load_manifest, merge, resolve

FIELDS = dict(id="a" * 16, set="flat", model="m", backend="diffusers", prompt="p", seed=0,
              width=8, height=8, steps=4, guidance=0.0, device="cpu")


def test_add_writes_file_and_manifest(data_root):
    s = Shard(data_root, "mac")
    rec = s.add(FIELDS, b"\x89PNG fake", "png")
    assert (s.dir / rec.file).read_bytes() == b"\x89PNG fake"
    assert rec.file == "images/flat/m/" + "a" * 16 + ".png"
    assert s.has("a" * 16) and len(s.records()) == 1
    assert len(rec.sha256) == 64 and rec.host == "mac"


def test_model_slug_is_filesystem_safe(data_root):
    rec = Shard(data_root, "mac").add({**FIELDS, "model": "black-forest-labs/FLUX.1-schnell"}, b"x", "png")
    assert "/FLUX" not in rec.file and "black-forest-labs__FLUX.1-schnell" in rec.file


def test_truncated_last_manifest_line_is_ignored(data_root):
    s = Shard(data_root, "mac")
    s.add(FIELDS, b"x", "png")
    with open(s.dir / "manifest.jsonl", "a") as f:
        f.write('{"id": "bbbb')                      # crash mid-append
    s2 = Shard(data_root, "mac")
    assert [r.id for r in s2.records()] == ["a" * 16]


def test_no_tmp_files_left_behind(data_root):
    s = Shard(data_root, "mac")
    s.add(FIELDS, b"x", "png")
    assert not list(s.dir.rglob("*.tmp"))


def test_merge_combines_shards_and_dedupes(data_root):
    Shard(data_root, "mac").add(FIELDS, b"same", "png")
    Shard(data_root, "host-b").add(FIELDS, b"same", "png")
    Shard(data_root, "host-b").add({**FIELDS, "id": "c" * 16}, b"other", "png")
    recs, warnings = merge(data_root)
    assert sorted(r.id for r in recs) == ["a" * 16, "c" * 16]
    assert warnings == []
    assert [r.id for r in load_manifest(data_root)] == [r.id for r in recs]
    for r in recs:
        assert resolve(data_root, r).exists()


def test_merge_warns_when_same_id_has_different_pixels(data_root):
    Shard(data_root, "mac").add(FIELDS, b"mps-output", "png")
    Shard(data_root, "pc4060").add(FIELDS, b"cuda-output", "png")
    recs, warnings = merge(data_root)
    assert len(recs) == 1
    assert len(warnings) == 1 and "a" * 16 in warnings[0]


def test_log_error_appends_jsonl(data_root):
    s = Shard(data_root, "mac")
    s.log_error("x" * 16, "boom")
    line = json.loads((s.dir / "errors.jsonl").read_text().splitlines()[0])
    assert line["id"] == "x" * 16 and line["message"] == "boom"


def test_append_after_truncated_line_is_not_lost(data_root):
    s = Shard(data_root, "mac")
    s.add(FIELDS, b"x", "png")
    with open(s.dir / "manifest.jsonl", "a") as f:
        f.write('{"id": "bbbb')                      # crash mid-append
    Shard(data_root, "mac").add({**FIELDS, "id": "c" * 16}, b"y", "png")   # resumed run
    assert [r.id for r in Shard(data_root, "mac").records()] == ["a" * 16, "c" * 16]


def test_merge_keeps_records_that_are_not_from_a_shard(data_root):
    # a downloaded data root (e.g. the Hugging Face dataset) has a manifest but no shards
    (data_root / "images").mkdir()
    (data_root / "images" / "x.png").write_bytes(b"downloaded")
    shared = {**FIELDS, "id": "d" * 16, "file": "images/x.png", "sha256": "0" * 64, "host": "export"}
    (data_root / "manifest.jsonl").write_text(json.dumps(shared) + "\n", encoding="utf-8")
    Shard(data_root, "mac").add(FIELDS, b"new", "png")
    recs, _ = merge(data_root)
    assert sorted(r.id for r in recs) == ["a" * 16, "d" * 16]
    assert {r.id: r.file for r in load_manifest(data_root)}["d" * 16] == "images/x.png"
