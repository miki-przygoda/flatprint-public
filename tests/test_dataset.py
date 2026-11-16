import io

import numpy as np
from PIL import Image

from flatprint.dataset import load_groups
from flatprint.prompts import flat_prompts
from flatprint.store import Shard, merge


def _png(colour, size=(4, 3)):
    b = io.BytesIO()
    Image.new("RGB", size, colour).save(b, "PNG")
    return b.getvalue()


def _add(shard, id, prompt, seed, colour, size=(4, 3), set_="flat-rgb"):
    shard.add({"id": id, "set": set_, "model": "m", "backend": "b", "prompt": prompt, "seed": seed,
               "width": size[0], "height": size[1]}, _png(colour, size), "png")


def test_groups_by_colour_with_target_rgb_and_stacked_pixels(data_root):
    dark, white = flat_prompts([{"name": "dark grey", "rgb": [64, 64, 64]}, {"name": "white", "rgb": [255, 255, 255]}])
    s = Shard(data_root, "mac")
    _add(s, "a", dark, 1, (60, 61, 62))
    _add(s, "b", dark, 2, (66, 66, 66))
    _add(s, "c", white, 3, (250, 250, 250))
    _add(s, "x", white, 4, (0, 0, 0), set_="other")
    merge(data_root)
    groups = load_groups(data_root, "flat-rgb")
    assert [g.name for g in groups] == ["dark grey", "white"]
    g = groups[0]
    assert g.target_rgb == (64, 64, 64) and g.seeds == [1, 2]
    assert g.pixels.shape == (2, 3, 4, 3) and g.pixels.dtype == np.uint8   # (n, height, width, channel)
    assert tuple(g.pixels[0, 0, 0]) == (60, 61, 62)


def test_groups_split_by_size(data_root):
    (dark,) = flat_prompts([{"name": "dark grey", "rgb": [64, 64, 64]}])
    s = Shard(data_root, "mac")
    _add(s, "a", dark, 1, (64, 64, 64), size=(4, 3))
    _add(s, "b", dark, 2, (64, 64, 64), size=(8, 8))
    merge(data_root)
    assert sorted(g.pixels.shape for g in load_groups(data_root, "flat-rgb")) == [(1, 3, 4, 3), (1, 8, 8, 3)]


def test_named_colour_prompts_get_their_name(data_root):
    (black,) = flat_prompts(["black"])
    s = Shard(data_root, "mac")
    _add(s, "a", black, 1, (5, 5, 5), set_="flat-pilot")
    merge(data_root)
    (g,) = load_groups(data_root, "flat-pilot")
    assert g.name == "black" and g.target_rgb == (0, 0, 0)


def test_read_ids_finds_store_ids_in_any_text(tmp_path):
    from flatprint.dataset import read_ids
    f = tmp_path / "flagged.md"
    f.write_text("| 1 | black | `852784b35510d0fe` | note |\n- id 6acc47d1468422a4 · seed 5\nnot-an-id 12345\n", encoding="utf-8")
    assert read_ids(f) == {"852784b35510d0fe", "6acc47d1468422a4"}


def test_load_groups_can_exclude_or_keep_only_ids(data_root):
    (dark,) = flat_prompts([{"name": "dark grey", "rgb": [64, 64, 64]}])
    s = Shard(data_root, "mac")
    _add(s, "aaaaaaaaaaaaaaaa", dark, 1, (64, 64, 64))
    _add(s, "bbbbbbbbbbbbbbbb", dark, 2, (64, 64, 64))
    merge(data_root)
    (g,) = load_groups(data_root, "flat-rgb", exclude={"aaaaaaaaaaaaaaaa"})
    assert g.seeds == [2]
    (g,) = load_groups(data_root, "flat-rgb", only={"aaaaaaaaaaaaaaaa"})
    assert g.seeds == [1]


def test_load_matrix_stacks_a_set_and_reuses_the_cache(data_root, monkeypatch):
    from flatprint import dataset
    dark, white = flat_prompts([{"name": "dark grey", "rgb": [64, 64, 64]}, "white"])
    s = Shard(data_root, "mac")
    _add(s, "aaaaaaaaaaaaaaaa", dark, 2, (60, 61, 62))
    _add(s, "bbbbbbbbbbbbbbbb", dark, 1, (66, 66, 66))
    _add(s, "cccccccccccccccc", white, 3, (250, 250, 250))
    merge(data_root)
    m = dataset.load_matrix(data_root, "flat-rgb")
    assert m.pixels.shape == (3, 3, 4, 3) and m.pixels.dtype == np.uint8
    assert [r.id for r in m.records] == ["bbbbbbbbbbbbbbbb", "aaaaaaaaaaaaaaaa", "cccccccccccccccc"]  # colour, then seed
    assert m.names == ["dark grey", "dark grey", "white"]
    assert tuple(m.pixels[1, 0, 0]) == (60, 61, 62)

    opened = []
    real_open = dataset.Image.open
    monkeypatch.setattr(dataset.Image, "open", lambda *a, **k: opened.append(1) or real_open(*a, **k))
    again = dataset.load_matrix(data_root, "flat-rgb")
    assert opened == [] and np.array_equal(again.pixels, m.pixels)


def test_load_matrix_rebuilds_when_the_set_changes(data_root):
    from flatprint.dataset import load_matrix
    (dark,) = flat_prompts([{"name": "dark grey", "rgb": [64, 64, 64]}])
    s = Shard(data_root, "mac")
    _add(s, "aaaaaaaaaaaaaaaa", dark, 1, (64, 64, 64))
    merge(data_root)
    assert len(load_matrix(data_root, "flat-rgb").records) == 1
    _add(Shard(data_root, "mac"), "bbbbbbbbbbbbbbbb", dark, 2, (64, 64, 64))
    merge(data_root)
    assert len(load_matrix(data_root, "flat-rgb").records) == 2


def test_load_matrix_needs_a_size_when_the_set_mixes_sizes(data_root):
    import pytest
    from flatprint.dataset import load_matrix
    (dark,) = flat_prompts([{"name": "dark grey", "rgb": [64, 64, 64]}])
    s = Shard(data_root, "mac")
    _add(s, "aaaaaaaaaaaaaaaa", dark, 1, (64, 64, 64), size=(4, 3))
    _add(s, "bbbbbbbbbbbbbbbb", dark, 2, (64, 64, 64), size=(8, 8))
    merge(data_root)
    with pytest.raises(ValueError, match="4x3"):
        load_matrix(data_root, "flat-rgb")
    assert load_matrix(data_root, "flat-rgb", size=(8, 8)).pixels.shape == (1, 8, 8, 3)


def test_load_matrix_cache_can_live_elsewhere(data_root, tmp_path, monkeypatch):
    from flatprint import dataset
    _add(Shard(data_root, "mac"), "aaaaaaaaaaaaaaaa", flat_prompts(["white"])[0], 1, (250, 250, 250))
    merge(data_root)
    monkeypatch.setenv("FLATPRINT_CACHE", str(tmp_path / "fast"))
    dataset.load_matrix(data_root, "flat-rgb")
    assert (tmp_path / "fast" / "flat-rgb-4x3.npy").exists() and not (data_root / "cache").exists()
