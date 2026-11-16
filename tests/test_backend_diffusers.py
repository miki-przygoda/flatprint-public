import io
from types import SimpleNamespace

import pytest
from PIL import Image

from flatprint.backends.base import encode_png, sniff_ext
from flatprint.backends.diffusers_local import DiffusersBackend, VaeRoundtripBackend, pick_dtype
from flatprint.config import GenJob

JOB = GenJob("flat", "org/m", "diffusers", "plain black", 7, 16, 8, 4, 0.0)


def test_sniff_ext():
    assert sniff_ext(b"\x89PNG\r\n\x1a\nrest") == "png"
    assert sniff_ext(b"\xff\xd8\xff\xe0rest") == "jpg"
    assert sniff_ext(b"RIFF\x00\x00\x00\x00WEBPrest") == "webp"
    assert sniff_ext(b"????") == "bin"


def test_encode_png_has_no_text_chunks(tiny_rgb):
    img = tiny_rgb()
    img.info["parameters"] = "should not be written"
    data = encode_png(img)
    assert sniff_ext(data) == "png"
    assert b"tEXt" not in data and b"iTXt" not in data and b"zTXt" not in data
    assert Image.open(io.BytesIO(data)).size == (32, 32)


def test_pick_dtype():
    assert pick_dtype("cuda", "auto", cuda_capability=(6, 1)) == "float16"   # 1080 Ti: no bf16
    assert pick_dtype("cuda", "auto", cuda_capability=(8, 9)) == "bfloat16"  # RTX 4060
    assert pick_dtype("mps", "auto") == "bfloat16"
    assert pick_dtype("cpu", "auto") == "float32"
    assert pick_dtype("cuda", "float16", cuda_capability=(8, 9)) == "float16"


class FakePipe:
    def __init__(self):
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(images=[Image.new("RGB", (kw["width"], kw["height"]))])


def test_diffusers_backend_passes_job_params_and_loads_once():
    pipe = FakePipe()
    loads = []
    b = DiffusersBackend("org/m", {"device": "cpu"},
                         pipeline_factory=lambda model, dtype: loads.append((model, dtype)) or pipe,
                         generator_factory=lambda seed: f"gen-{seed}")
    out = b.generate(JOB)
    b.generate(JOB)
    assert loads == [("org/m", "float32")]
    kw = pipe.calls[0]
    assert kw["prompt"] == "plain black" and kw["width"] == 16 and kw["height"] == 8
    assert kw["num_inference_steps"] == 4 and kw["guidance_scale"] == 0.0
    assert kw["generator"] == "gen-7"
    assert out.ext == "png" and out.device == "cpu"
    assert Image.open(io.BytesIO(out.data)).size == (16, 8)


def test_diffusers_backend_omits_unset_steps_and_guidance():
    pipe = FakePipe()
    b = DiffusersBackend("org/m", {"device": "cpu"}, pipeline_factory=lambda m, d: pipe,
                         generator_factory=lambda s: None)
    b.generate(GenJob("flat", "org/m", "diffusers", "p", 0, 8, 8, None, None))
    assert "num_inference_steps" not in pipe.calls[0] and "guidance_scale" not in pipe.calls[0]


def test_vae_roundtrip_uses_flat_colour_from_prompt():
    seen = []

    def fake_roundtrip(img):
        seen.append(img.getpixel((0, 0)))
        return img

    b = VaeRoundtripBackend("org/m", {"device": "cpu"}, vae_factory=lambda model, dtype: fake_roundtrip)
    out = b.generate(GenJob("vae-flat", "org/m", "vae-roundtrip", "white", 0, 8, 8, None, None))
    assert seen == [(255, 255, 255)] and out.ext == "png"


def test_outputs_record_provenance():
    b = DiffusersBackend("org/m", {"device": "cpu"}, pipeline_factory=lambda m, d: FakePipe(),
                         generator_factory=lambda s: None)
    extra = b.generate(JOB).extra
    assert extra["dtype"] == "float32"
    assert {"torch", "diffusers", "platform", "gpu"} <= set(extra)
    assert extra["gpu"] is None                                   # cpu device
    v = VaeRoundtripBackend("org/m", {"device": "cpu"}, vae_factory=lambda m, d: (lambda im: im))
    assert {"torch", "diffusers", "platform", "gpu"} <= set(
        v.generate(GenJob("vae-flat", "org/m", "vae-roundtrip", "black", 0, 8, 8, None, None)).extra)


def _decoded(out):
    import io
    import numpy as np
    return np.asarray(Image.open(io.BytesIO(out.data)).convert("RGB")).astype(float)


def test_vae_roundtrip_can_add_seeded_pixel_noise_before_encoding():
    ident = lambda m, d: (lambda im: im)                       # noqa: E731  (shows what went into the VAE)
    b = VaeRoundtripBackend("org/m", {"device": "cpu", "noise_std": 2.0}, vae_factory=ident)
    job = lambda seed: GenJob("vae-noise", "org/m", "vae-roundtrip", "RGB(8, 8, 8)", seed, 64, 64, None, None)  # noqa: E731
    a, again, other = (_decoded(b.generate(job(s))) for s in (1, 1, 2))
    assert (a == again).all() and not (a == other).all()       # seeded
    assert abs(a.mean() - 8) < 0.3 and 1.7 < a.std() < 2.3
    flat = VaeRoundtripBackend("org/m", {"device": "cpu"}, vae_factory=ident)
    assert _decoded(flat.generate(job(1))).std() == 0           # no option: unchanged behaviour


def test_vae_random_latents_decode_without_encoding():
    calls = []

    def fake_decode(seed, width, height):
        calls.append((seed, width, height))
        return Image.new("RGB", (width, height), (seed, 0, 0))

    b = VaeRoundtripBackend("org/m", {"device": "cpu", "latent": "random"}, vae_factory=lambda m, d: 1 / 0,
                            decode_factory=lambda m, d: fake_decode)
    out = b.generate(GenJob("vae-latent", "org/m", "vae-roundtrip", "black", 7, 32, 16, None, None))
    assert calls == [(7, 32, 16)] and _decoded(out)[0, 0, 0] == 7


def test_vae_unknown_latent_option_is_refused():
    with pytest.raises(ValueError, match="latent"):
        VaeRoundtripBackend("org/m", {"device": "cpu", "latent": "nope"})


def test_vae_flat_latent_plus_noise_decodes_seeded_latent_noise_around_a_flat_colour():
    calls = []

    def fake(img, seed, latent_noise):
        calls.append((img.getpixel((0, 0)), img.size, seed, latent_noise))
        return img

    b = VaeRoundtripBackend("org/m", {"device": "cpu", "latent": "flat+noise", "latent_noise": 0.1},
                            vae_factory=lambda m, d: 1 / 0, latent_factory=lambda m, d: fake)
    out = b.generate(GenJob("vae-lf", "org/m", "vae-roundtrip", "RGB(8, 8, 8)", 5, 32, 16, None, None))
    assert calls == [((8, 8, 8), (32, 16), 5, 0.1)]               # the flat colour, never a noisy image
    assert out.extra["vae_input"] == "flat+noise" and out.extra["latent_noise"] == 0.1


def test_vae_flat_latent_plus_noise_needs_a_noise_level():
    with pytest.raises(ValueError, match="latent_noise"):
        VaeRoundtripBackend("org/m", {"device": "cpu", "latent": "flat+noise"})
