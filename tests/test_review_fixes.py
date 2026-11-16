"""Regression tests for the final-review findings (I1–I7)."""

import io
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flatprint.backends.base import GenOutput
from flatprint.backends.diffusers_local import DiffusersBackend, VaeRoundtripBackend
from flatprint.config import ConfigError, GenJob, expand_jobs, load_config
from flatprint.derive import derive_set
from flatprint.ops import apply_ops, parse_ops
from flatprint.runner import run
from flatprint.store import Shard, load_manifest, merge

SRC = Path(__file__).parent.parent / "python" / "flatprint"


def png(w, h, colour=(0, 0, 0)):
    b = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(b, "PNG")
    return b.getvalue()


# I1: every text-mode file operation names its encoding (Windows defaults to cp1252)
def test_all_text_io_is_explicit_utf8():
    offenders = []
    for f in SRC.rglob("*.py"):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            text_io = re.search(r"read_text\(|write_text\(|(?<![.\w])open\(", line)
            binary = re.search(r"[\"'](rb|wb|ab)\+?[\"']", line)
            if text_io and not binary and "encoding=" not in line:
                offenders.append(f"{f.name}:{n}: {line.strip()}")
    assert offenders == []


def test_non_ascii_caption_roundtrips_through_store_and_merge(data_root):
    shard = Shard(data_root, "pc4060")
    shard.add({"id": "a" * 16, "set": "content", "model": "m", "backend": "x",
               "prompt": "a café in Kraków — ünïcode"}, png(4, 4), "png")
    recs, _ = merge(data_root)
    assert recs[0].prompt == "a café in Kraków — ünïcode"


# I2: vae-roundtrip reachable from a config with kind="flat"
def test_vae_roundtrip_works_from_flat_config(tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text('name="v"\nset="vae-flat"\nbackend="vae-roundtrip"\nmodel="org/m"\nsizes=[[8,8]]\n'
                   '[seeds]\nstart=0\nper_prompt=1\n[prompts]\nkind="flat"\ncolours=["white"]\n', encoding="utf-8")
    job = expand_jobs(load_config(cfg))[0]
    seen = []
    b = VaeRoundtripBackend("org/m", {"device": "cpu"}, vae_factory=lambda m, d: (lambda im: seen.append(im.getpixel((0, 0))) or im))
    b.generate(job)
    assert seen == [(255, 255, 255)]


def test_vae_roundtrip_rejects_caption_prompts(tmp_path):
    caps = tmp_path / "c.txt"
    caps.write_text("a cat\n", encoding="utf-8")
    cfg = tmp_path / "v.toml"
    cfg.write_text(f'name="v"\nset="s"\nbackend="vae-roundtrip"\nmodel="m"\nsizes=[[8,8]]\n'
                   f'[seeds]\nstart=0\n[prompts]\nkind="captions"\nfile="{caps.name}"\nn=1\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="vae-roundtrip"):
        load_config(cfg)


# I3: non-finite pipeline output is an error, not a silently black image
class NpPipe:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self

    def __call__(self, **kw):
        assert kw["output_type"] == "np"
        return SimpleNamespace(images=np.full((1, kw["height"], kw["width"], 3), self.value, dtype=np.float32))


def test_nan_output_raises():
    b = DiffusersBackend("m", {"device": "cpu"}, pipeline_factory=lambda m, d: NpPipe(np.nan),
                         generator_factory=lambda s: None)
    with pytest.raises(ValueError, match="non-finite"):
        b.generate(GenJob("flat", "m", "diffusers", "p", 0, 8, 8, None, None))


def test_np_output_converted_like_diffusers():
    b = DiffusersBackend("m", {"device": "cpu"}, pipeline_factory=lambda m, d: NpPipe(0.5),
                         generator_factory=lambda s: None)
    out = b.generate(GenJob("flat", "m", "diffusers", "p", 0, 8, 4, None, None))
    img = Image.open(io.BytesIO(out.data))
    assert img.size == (8, 4) and img.getpixel((0, 0)) == (128, 128, 128)


# I4: bad [options] fail at config load, before any model loads
@pytest.mark.parametrize("opts,word", [('dtype="fp16"', "dtype"), ('offload="full"', "offload")])
def test_bad_options_fail_fast(tmp_path, opts, word):
    cfg = tmp_path / "o.toml"
    cfg.write_text('name="o"\nset="s"\nbackend="diffusers"\nmodel="m"\nsizes=[[8,8]]\n[seeds]\nstart=0\n'
                   f'[prompts]\nkind="flat"\ncolours=["black"]\n[options]\n{opts}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=word):
        load_config(cfg)


# I5: stored size is the real image size; the request is kept in extra
class SizedBackend:
    name = "x"

    def generate(self, job):
        return GenOutput(png(10, 6), "png", "api")


class GarbageBackend:
    name = "x"

    def generate(self, job):
        return GenOutput(b"not an image", "bin", "api")


def test_runner_records_actual_size(data_root):
    shard = Shard(data_root, "mac")
    run([GenJob("s", "m", "x", "p", 0, 8, 8, None, None)], shard, SizedBackend())
    r = shard.records()[0]
    assert (r.width, r.height) == (10, 6) and r.extra["requested_size"] == [8, 8]


def test_runner_rejects_undecodable_output(data_root):
    shard = Shard(data_root, "mac")
    s = run([GenJob("s", "m", "x", "p", 0, 8, 8, None, None)], shard, GarbageBackend())
    assert (s.done, s.failed) == (0, 1) and shard.records() == []


# I6: derived ids depend on the target set and on the normalised ops
def test_same_ops_into_two_sets_both_materialise(data_root):
    Shard(data_root, "mac").add({"id": "a" * 16, "set": "c", "model": "m", "backend": "x"}, png(8, 8), "png")
    merge(data_root)
    out = Shard(data_root, "mac")
    assert derive_set(load_manifest(data_root), data_root, out, "jpeg:90", "set-one").done == 1
    assert derive_set(load_manifest(data_root), data_root, out, "jpeg:90", "set-two").done == 1


def test_ops_spelling_does_not_change_id(data_root):
    Shard(data_root, "mac").add({"id": "a" * 16, "set": "c", "model": "m", "backend": "x"}, png(8, 8), "png")
    merge(data_root)
    out = Shard(data_root, "mac")
    derive_set(load_manifest(data_root), data_root, out, "jpeg:90", "q")
    assert derive_set(load_manifest(data_root), data_root, out, " jpeg:90 ", "q").skipped == 1


# I7: derived outputs carry no source metadata (ICC)
def test_png_output_strips_icc():
    b = io.BytesIO()
    Image.new("RGB", (8, 8)).save(b, "JPEG", icc_profile=b"\x00" * 128)
    out = apply_ops(b.getvalue(), parse_ops("png"))
    assert b"iCCP" not in out


def test_vae_float_conversion_rejects_nan():
    from flatprint.backends.diffusers_local import _signed_to_pil
    with pytest.raises(ValueError, match="non-finite"):
        _signed_to_pil(np.full((4, 4, 3), np.nan, dtype=np.float32), "vae")
    assert _signed_to_pil(np.zeros((4, 4, 3), dtype=np.float32), "vae").getpixel((0, 0)) == (128, 128, 128)
