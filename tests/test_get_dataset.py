import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "get_dataset.py"
spec = importlib.util.spec_from_file_location("get_dataset", SCRIPT)
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)


def test_groups_cover_each_set_once():
    sets = [s for members in gd.GROUPS.values() for s in members]
    assert len(sets) == len(set(sets)) == 34


def test_wanted_sets_expands_groups_and_rejects_unknown_ones():
    assert gd.wanted_sets(["real"], ["showcase"]) == ["phone-blank", "real-phone", "real-phone-dim", "showcase"]
    assert len(gd.wanted_sets(["all"], [])) == 34
    with pytest.raises(SystemExit):
        gd.wanted_sets(["nope"], [])


def _release(tmp_path):
    """A tiny dataset release: manifest + one archive per set, as on Hugging Face."""
    rel = tmp_path / "release"
    (rel / "archives").mkdir(parents=True)
    lines = []
    for s, payload in (("real", b"phone"), ("scenes", b"scene")):
        name = f"images/{s}/{s}.png"
        with tarfile.open(rel / "archives" / f"{s}.tar", "w") as t:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            t.addfile(info, io.BytesIO(payload))
        lines.append(json.dumps({"id": s * 2, "set": s, "model": "m", "backend": "x", "host": "export", "file": name,
                                 "sha256": hashlib.sha256(payload).hexdigest()}))
    (rel / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return rel


def test_extract_one_set_from_local_archives_and_verify(tmp_path, capsys):
    rel, dest = _release(tmp_path), tmp_path / "data"
    assert gd.main(["--sets", "real", "--archives", str(rel / "archives"), "--dest", str(dest)]) == 0
    assert (dest / "images" / "real" / "real.png").read_bytes() == b"phone"
    assert not (dest / "images" / "scenes").exists()
    assert (rel / "archives" / "real.tar").exists()          # someone else's archives are never deleted
    out = capsys.readouterr().out
    assert "1 checked: 1 identical" in out and "skipped 1 set" in out


def test_missing_archive_is_an_error(tmp_path):
    rel = _release(tmp_path)
    with pytest.raises(SystemExit):
        gd.main(["--sets", "nope", "--archives", str(rel / "archives"), "--dest", str(tmp_path / "d")])
