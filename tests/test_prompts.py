import json

import pytest

from flatprint.prompts import caption_prompts, flat_prompts


def test_flat_prompts_one_per_colour():
    ps = flat_prompts(["black", "white"])
    assert len(ps) == 2
    assert "black" in ps[0] and "white" in ps[1]


def test_flat_prompts_rejects_unknown_colour():
    with pytest.raises(ValueError, match="mauve"):
        flat_prompts(["mauve"])


def test_caption_prompts_from_coco_json_is_deterministic(tmp_path):
    ann = {"annotations": [{"image_id": i, "caption": f" A photo number {i}. "} for i in range(50)]}
    f = tmp_path / "captions.json"
    f.write_text(json.dumps(ann))
    a = caption_prompts(f, n=10, seed=0)
    b = caption_prompts(f, n=10, seed=0)
    assert a == b and len(a) == 10
    assert all(p == p.strip() for p in a)
    assert len(set(a)) == 10


def test_caption_prompts_from_txt(tmp_path):
    f = tmp_path / "captions.txt"
    f.write_text("a cat\n\na dog\na bird\n")
    assert sorted(caption_prompts(f, n=3, seed=1)) == ["a bird", "a cat", "a dog"]


def test_caption_prompts_too_few(tmp_path):
    f = tmp_path / "captions.txt"
    f.write_text("a cat\n")
    with pytest.raises(ValueError, match="only 1"):
        caption_prompts(f, n=5, seed=0)


def test_caption_prompts_can_exclude_words_before_sampling(tmp_path):
    f = tmp_path / "captions.txt"
    f.write_text("a cat on a Wii\na dog\na bird\nA Volvo bus\nvolvox algae\n")
    got = caption_prompts(f, n=3, seed=0, exclude=["wii", "Volvo"])
    assert sorted(got) == ["a bird", "a dog", "volvox algae"]     # whole words only, any case
    with pytest.raises(ValueError, match="only 3"):
        caption_prompts(f, n=4, seed=0, exclude=["wii", "volvo"])


def test_config_passes_caption_exclusions(tmp_path):
    from flatprint.config import load_config, expand_jobs
    (tmp_path / "caps.txt").write_text("a Yamaha bike\nYAMAHA piano\na yamaha keyboard\na red apple on a table\na tree\n")
    cfg = tmp_path / "c.toml"
    cfg.write_text('name = "c"\nset = "s"\nbackend = "diffusers"\nmodel = "org/m"\nsizes = [[8, 8]]\n'
                   '[seeds]\nstart = 0\nper_prompt = 1\n'
                   '[prompts]\nkind = "captions"\nfile = "caps.txt"\nn = 2\nseed = 1\nexclude = ["yamaha"]\n')   # seed 1 alone would pick a Yamaha one
    assert sorted(j.prompt for j in expand_jobs(load_config(cfg))) == ["a red apple on a table", "a tree"]


def test_excluding_a_word_only_replaces_the_captions_that_hold_it(tmp_path):
    f = tmp_path / "captions.txt"
    f.write_text("\n".join([f"a plain caption {i}" for i in range(40)] + [f"a pepsi caption {i}" for i in range(10)]))
    a = caption_prompts(f, n=20, seed=3)
    b = caption_prompts(f, n=20, seed=3, exclude=["pepsi"])
    kept = [c for c in a if "pepsi" not in c]
    assert b[:len(kept)] == kept                                        # the clean ones stay, in order
    assert len(b) == 20 and not any("pepsi" in c for c in b)
    assert not set(b[len(kept):]) & set(a)                              # the gaps are filled by new ones only
