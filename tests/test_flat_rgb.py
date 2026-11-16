import pytest

from flatprint.backends.diffusers_local import VaeRoundtripBackend
from flatprint.config import ConfigError, expand_jobs, load_config
from flatprint.prompts import flat_prompts, rgb_of_prompt

CFG = """
name = "t"
set = "flat-rgb"
backend = "{backend}"
model = "org/m"
sizes = [[8, 8]]
[seeds]
{seeds}
[prompts]
kind = "flat"
colours = [{{ name = "dark grey", rgb = [64, 64, 64] }}, "black"]
"""


def cfg(tmp_path, seeds="start = 0\nper_prompt = 5", backend="diffusers"):
    p = tmp_path / "c.toml"
    p.write_text(CFG.format(seeds=seeds, backend=backend), encoding="utf-8")
    return p


def test_rgb_colour_puts_exact_value_in_prompt():
    (p,) = flat_prompts([{"name": "dark grey", "rgb": [64, 64, 64]}])
    assert "RGB(64, 64, 64)" in p and "#404040" in p and "dark grey" in p
    assert rgb_of_prompt(p) == (64, 64, 64)


def test_rgb_colour_must_be_three_bytes():
    with pytest.raises(ValueError, match="rgb"):
        flat_prompts([{"name": "bad", "rgb": [300, 0, 0]}])
    with pytest.raises(ValueError, match="rgb"):
        flat_prompts([{"name": "bad", "rgb": [1, 2]}])


def test_named_colours_still_work_alongside_rgb(tmp_path):
    c = load_config(cfg(tmp_path))
    assert len(c.prompts) == 2 and "RGB(64, 64, 64)" in c.prompts[0] and "black" in c.prompts[1]


def test_random_seeds_are_unique_reproducible_and_not_sequential(tmp_path):
    seeds = "random = true\npick_seed = 7\nper_prompt = 5"
    a = [j.seed for j in expand_jobs(load_config(cfg(tmp_path, seeds)))]
    b = [j.seed for j in expand_jobs(load_config(cfg(tmp_path, seeds)))]
    assert a == b and len(set(a)) == 10
    assert a != list(range(a[0], a[0] + 10))
    other = [j.seed for j in expand_jobs(load_config(cfg(tmp_path, "random = true\npick_seed = 8\nper_prompt = 5")))]
    assert other != a


def test_random_seeds_need_pick_seed(tmp_path):
    with pytest.raises(ConfigError, match="pick_seed"):
        load_config(cfg(tmp_path, "random = true\nper_prompt = 5"))


def test_vae_roundtrip_uses_exact_rgb(tmp_path):
    job = expand_jobs(load_config(cfg(tmp_path, backend="vae-roundtrip")))[0]
    seen = []
    b = VaeRoundtripBackend("org/m", {"device": "cpu"}, vae_factory=lambda m, d: (lambda im: seen.append(im.getpixel((0, 0))) or im))
    b.generate(job)
    assert seen == [(64, 64, 64)]
