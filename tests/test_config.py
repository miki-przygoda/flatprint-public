import pytest

from flatprint.config import ConfigError, GenJob, expand_jobs, load_config

FLAT = """
name = "t-flat"
set = "flat"
backend = "diffusers"
model = "org/model"
steps = 4
guidance = 0.0
sizes = [[64, 64], [32, 32]]

[seeds]
start = 100
per_prompt = 3

[prompts]
kind = "flat"
colours = ["black", "white"]
"""


def write(tmp_path, text, name="c.toml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_and_expand_flat(tmp_path):
    cfg = load_config(write(tmp_path, FLAT))
    jobs = expand_jobs(cfg)
    assert len(jobs) == 2 * 2 * 3                      # colours x sizes x per_prompt
    assert len({j.key() for j in jobs}) == len(jobs)   # unique ids
    assert {j.seed for j in jobs} == set(range(100, 106))
    assert all(isinstance(j, GenJob) and j.set == "flat" for j in jobs)


def test_job_key_is_deterministic_and_16_hex():
    j = GenJob("flat", "org/model", "diffusers", "p", 1, 64, 64, 4, 0.0)
    assert j.key() == GenJob("flat", "org/model", "diffusers", "p", 1, 64, 64, 4, 0.0).key()
    assert len(j.key()) == 16 and int(j.key(), 16) >= 0
    assert j.key() != GenJob("flat", "org/model", "diffusers", "p", 2, 64, 64, 4, 0.0).key()


def test_unknown_top_level_key_fails_fast(tmp_path):
    with pytest.raises(ConfigError, match="stepz"):
        load_config(write(tmp_path, FLAT.replace("steps = 4", "stepz = 4")))


def test_unknown_backend_fails_fast(tmp_path):
    with pytest.raises(ConfigError, match="backend"):
        load_config(write(tmp_path, FLAT.replace('"diffusers"', '"dall-e-9"')))


def test_missing_captions_file_fails_fast(tmp_path):
    text = FLAT.replace('kind = "flat"\ncolours = ["black", "white"]',
                        'kind = "captions"\nfile = "nope.json"\nn = 5\nseed = 0')
    with pytest.raises(ConfigError, match="nope.json"):
        load_config(write(tmp_path, text))
