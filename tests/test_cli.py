from flatprint import cli

CFG = """
name = "t"
set = "flat"
backend = "diffusers"
model = "org/m"
sizes = [[8, 8]]
[seeds]
start = 0
per_prompt = 2
[prompts]
kind = "flat"
colours = ["black"]
"""


def test_generate_merge_info(tmp_path, data_root, monkeypatch, capsys):
    from flatprint.backends.base import GenOutput

    class Fake:
        name = "diffusers"

        def generate(self, job):
            import io
            from PIL import Image
            b = io.BytesIO()
            Image.new("RGB", (8, 8), (job.seed, 0, 0)).save(b, "PNG")
            return GenOutput(b.getvalue(), "png", "cpu")

    monkeypatch.setattr(cli, "get_backend", lambda name, model, options: Fake())
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG)
    assert cli.main(["--data", str(data_root), "generate", str(cfg), "--host", "mac"]) == 0
    assert cli.main(["--data", str(data_root), "merge"]) == 0
    assert cli.main(["--data", str(data_root), "info"]) == 0
    out = capsys.readouterr().out
    assert "done=2" in out and "flat" in out and "org/m" in out


def test_generate_bad_config_fails_before_backend(tmp_path, data_root, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(cli, "get_backend", lambda *a: called.append(a))
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG.replace('"diffusers"', '"nope"'))
    assert cli.main(["--data", str(data_root), "generate", str(cfg)]) == 2
    assert called == [] and "backend" in capsys.readouterr().err


def fake_backend(monkeypatch):
    import io
    from PIL import Image
    from flatprint.backends.base import GenOutput

    class Fake:
        name = "diffusers"

        def generate(self, job):
            b = io.BytesIO()
            Image.new("RGB", (job.width, job.height), (200, 10, 10)).save(b, "PNG")
            return GenOutput(b.getvalue(), "png", "cpu")

    monkeypatch.setattr(cli, "get_backend", lambda name, model, options: Fake())


def test_prompt_makes_one_adhoc_image(tmp_path, data_root, monkeypatch, capsys):
    from flatprint.store import Shard
    fake_backend(monkeypatch)
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG, encoding="utf-8")
    assert cli.main(["--data", str(data_root), "prompt", "a red fox in snow", "--config", str(cfg),
                     "--seed", "5", "--size", "16x8", "--host", "mac"]) == 0
    [r] = Shard(data_root, "mac").records()
    assert (r.set, r.prompt, r.seed, r.width, r.height, r.model) == ("adhoc", "a red fox in snow", 5, 16, 8, "org/m")
    assert r.id in capsys.readouterr().out


def test_prompt_without_seed_records_a_random_one(tmp_path, data_root, monkeypatch):
    from flatprint.store import Shard
    fake_backend(monkeypatch)
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG, encoding="utf-8")
    cli.main(["--data", str(data_root), "prompt", "a fox", "--config", str(cfg), "--host", "mac"])
    cli.main(["--data", str(data_root), "prompt", "a fox", "--config", str(cfg), "--host", "mac"])
    seeds = [r.seed for r in Shard(data_root, "mac").records()]
    assert len(seeds) == 2 and all(isinstance(s, int) for s in seeds) and seeds[0] != seeds[1]


def test_merge_writes_browsable_index_csv(tmp_path, data_root, monkeypatch):
    import csv
    fake_backend(monkeypatch)
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG, encoding="utf-8")
    cli.main(["--data", str(data_root), "prompt", "a fox", "--config", str(cfg), "--seed", "1", "--host", "mac"])
    cli.main(["--data", str(data_root), "merge"])
    rows = list(csv.DictReader(open(data_root / "index.csv", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["set"] == "adhoc" and rows[0]["prompt"] == "a fox" and rows[0]["seed"] == "1"
    assert (data_root / rows[0]["file"]).exists()


def test_export_copies_set_with_readable_names(tmp_path, data_root, monkeypatch):
    fake_backend(monkeypatch)
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG, encoding="utf-8")
    for seed in (7, 8):
        cli.main(["--data", str(data_root), "prompt", "Candid street portrait of a woman, rain!", "--config", str(cfg),
                  "--seed", str(seed), "--set", "showcase", "--host", "mac"])
    cli.main(["--data", str(data_root), "prompt", "a fox", "--config", str(cfg), "--seed", "1", "--host", "mac"])
    cli.main(["--data", str(data_root), "merge"])
    out = tmp_path / "preview"
    assert cli.main(["--data", str(data_root), "export", "--set", "showcase", "--to", str(out)]) == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png"))
    assert files == ["01-candid-street-portrait-of-a-woman-rain/seed-7.png",
                     "01-candid-street-portrait-of-a-woman-rain/seed-8.png"]
    assert "Candid street portrait of a woman, rain!" in (out / "prompts.txt").read_text(encoding="utf-8")


def test_export_keeps_same_seed_at_two_sizes(tmp_path, data_root, monkeypatch):
    fake_backend(monkeypatch)
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG.replace("sizes = [[8, 8]]", "sizes = [[8, 8], [16, 8]]"), encoding="utf-8")
    cli.main(["--data", str(data_root), "generate", str(cfg), "--host", "mac"])
    cli.main(["--data", str(data_root), "merge"])
    out = tmp_path / "preview"
    assert cli.main(["--data", str(data_root), "export", "--set", "flat", "--to", str(out)]) == 0
    names = sorted(p.name for p in out.rglob("*.png"))
    assert names == ["seed-0-16x8.png", "seed-0-8x8.png", "seed-1-16x8.png", "seed-1-8x8.png"]


def test_export_folders_differ_when_prompts_share_a_long_start(tmp_path, data_root, monkeypatch):
    fake_backend(monkeypatch)
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG, encoding="utf-8")
    start = "a plain solid flat image filled with one exact colour, "
    for p in (start + "RGB(0, 0, 0), black", start + "RGB(64, 64, 64), dark grey"):
        cli.main(["--data", str(data_root), "prompt", p, "--config", str(cfg), "--seed", "1", "--set", "s", "--host", "mac"])
    cli.main(["--data", str(data_root), "merge"])
    out = tmp_path / "preview"
    cli.main(["--data", str(data_root), "export", "--set", "s", "--to", str(out)])
    assert sorted(d.name for d in out.iterdir() if d.is_dir()) == ["01-rgb-0-0-0-black", "02-rgb-64-64-64-dark-grey"]


def test_generate_per_prompt_override_and_hours(tmp_path, data_root, monkeypatch, capsys):
    fake_backend(monkeypatch)
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG, encoding="utf-8")
    assert cli.main(["--data", str(data_root), "generate", str(cfg), "--host", "mac",
                     "--per-prompt", "4", "--hours", "1"]) == 0
    assert "done=4" in capsys.readouterr().out


def _stored(data_root, n=3):
    import io
    from PIL import Image
    from flatprint.store import Shard
    s = Shard(data_root, "mac")
    for i in range(n):
        b = io.BytesIO()
        Image.new("RGB", (4, 4), (i, 0, 0)).save(b, "PNG")
        s.add({"id": f"{i:016x}", "set": "a" if i else "b", "model": "m", "backend": "x", "width": 4, "height": 4},
              b.getvalue(), "png")
    assert cli.main(["--data", str(data_root), "merge"]) == 0


def test_verify_passes_when_every_file_matches(data_root, capsys):
    _stored(data_root)
    assert cli.main(["--data", str(data_root), "verify"]) == 0
    assert "3 checked: 3 identical, 0 differ, 0 missing" in capsys.readouterr().out


def test_verify_fails_on_a_changed_or_missing_file(data_root, capsys):
    from flatprint.store import load_manifest
    _stored(data_root)
    recs = load_manifest(data_root)
    (data_root / recs[0].file).write_bytes(b"changed")
    (data_root / recs[1].file).unlink()
    assert cli.main(["--data", str(data_root), "verify"]) == 1
    out = capsys.readouterr().out
    assert "1 identical, 1 differ, 1 missing" in out and recs[0].id in out and recs[1].id in out


def test_verify_can_check_one_set(data_root, capsys):
    _stored(data_root)
    assert cli.main(["--data", str(data_root), "verify", "--set", "a"]) == 0
    assert "2 checked" in capsys.readouterr().out


def test_verify_fails_when_there_is_no_manifest(data_root, capsys):
    assert cli.main(["--data", str(data_root), "verify"]) == 1
    assert "manifest" in capsys.readouterr().err


def test_verify_present_skips_sets_with_no_files_on_disk(data_root, capsys):
    from flatprint.store import load_manifest
    _stored(data_root)
    for r in load_manifest(data_root):
        if r.set == "b":
            (data_root / r.file).unlink()                  # set b not downloaded
    assert cli.main(["--data", str(data_root), "verify", "--present"]) == 0
    out = capsys.readouterr().out
    assert "2 checked: 2 identical" in out and "skipped 1 set" in out and "b" in out


def test_verify_present_still_fails_on_a_partly_missing_set(data_root, capsys):
    from flatprint.store import load_manifest
    _stored(data_root)
    a = [r for r in load_manifest(data_root) if r.set == "a"]
    (data_root / a[0].file).unlink()
    assert cli.main(["--data", str(data_root), "verify", "--present"]) == 1


def test_verify_present_fails_when_no_set_is_on_disk(data_root, capsys):
    from flatprint.store import load_manifest
    _stored(data_root)
    for r in load_manifest(data_root):
        (data_root / r.file).unlink()
    assert cli.main(["--data", str(data_root), "verify", "--present"]) == 1
