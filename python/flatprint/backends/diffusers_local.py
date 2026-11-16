"""Local diffusers text-to-image backend, and a VAE round-trip backend for flat images."""

from PIL import Image

from flatprint.backends.base import GenOutput, encode_png
from flatprint.config import GenJob
from flatprint.prompts import FLAT_COLOURS, FLAT_RGB, rgb_of_prompt

COLOUR_OF_PROMPT = {sentence: colour for colour, sentence in FLAT_COLOURS.items()}


def provenance(device: str) -> dict:
    """Library versions, platform and GPU name, so a merge mismatch can be traced to its cause."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    gpu = None
    if device == "cuda":
        import torch
        gpu = torch.cuda.get_device_name()
    elif device == "mps":
        gpu = f"Apple {platform.machine()} (MPS)"
    return {"torch": ver("torch"), "diffusers": ver("diffusers"), "platform": platform.platform(), "gpu": gpu}


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str = "auto", cuda_capability: tuple[int, int] | None = None) -> str:
    if requested != "auto":
        return requested
    if device == "cuda":
        if cuda_capability is None:
            import torch
            cuda_capability = torch.cuda.get_device_capability()
        return "bfloat16" if cuda_capability >= (8, 0) else "float16"
    if device == "mps":
        return "bfloat16"
    return "float32"


def _torch_dtype(name: str):
    import torch
    return getattr(torch, name)


def _default_pipeline_factory(model: str, dtype: str):
    from diffusers import DiffusionPipeline
    return DiffusionPipeline.from_pretrained(model, dtype=_torch_dtype(dtype))


def _default_generator_factory(seed: int):
    import torch
    return torch.Generator("cpu").manual_seed(seed)  # same latents on every device


class DiffusersBackend:
    name = "diffusers"

    def __init__(self, model: str, options: dict, pipeline_factory=None, generator_factory=None):
        self.model = model
        self.options = options
        self.device = options.get("device") or pick_device()
        self.dtype = pick_dtype(self.device, options.get("dtype", "auto"))
        self._factory = pipeline_factory or _default_pipeline_factory
        self._gen = generator_factory or _default_generator_factory
        self._pipe = None
        self._prov = provenance(self.device)

    def _load(self):
        pipe = self._factory(self.model, self.dtype)
        offload = self.options.get("offload", "none")
        if offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        return pipe

    def generate(self, job: GenJob) -> GenOutput:
        if self._pipe is None:
            self._pipe = self._load()
        kw = dict(prompt=job.prompt, width=job.width, height=job.height, generator=self._gen(job.seed))
        if job.steps is not None:
            kw["num_inference_steps"] = job.steps
        if job.guidance is not None:
            kw["guidance_scale"] = job.guidance
        kw.update(self.options.get("call_kwargs", {}))
        kw["output_type"] = "np"  # floats, so NaN/inf (fp16 overflow) is caught, not saved as black
        image = _to_pil(self._pipe(**kw).images[0], job)
        return GenOutput(encode_png(image), "png", self.device, {"dtype": self.dtype, **self._prov})


def _to_pil(image, job: GenJob) -> Image.Image:
    """Convert a pipeline's float [0, 1] output exactly as diffusers' numpy_to_pil does."""
    if isinstance(image, Image.Image):
        return image
    import numpy as np
    arr = np.asarray(image)
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite pixels from the pipeline for job {job.key()} (fp16 overflow?)")
    return Image.fromarray((arr * 255).round().astype("uint8"))


def _default_vae_factory(model: str, dtype: str, device: str):
    import numpy as np
    import torch
    from diffusers import AutoModel

    # AutoModel picks the model's own VAE class (AutoencoderKL for FLUX.1, AutoencoderKLFlux2 for FLUX.2, ...)
    vae = AutoModel.from_pretrained(model, subfolder="vae", dtype=_torch_dtype(dtype))
    if dtype == "float16" and getattr(vae.config, "force_upcast", False):
        vae = vae.to(torch.float32)  # e.g. the SDXL VAE overflows in fp16
    vae = vae.to(device)

    @torch.no_grad()
    def roundtrip(img: Image.Image) -> Image.Image:
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
        z = vae.encode(x.to(device, vae.dtype)).latent_dist.mode()
        y = vae.decode(z).sample[0].float()
        return _signed_to_pil(y.permute(1, 2, 0).cpu().numpy(), model)

    return roundtrip


def _default_decode_factory(model: str, dtype: str, device: str):
    import torch
    from diffusers import AutoModel

    vae = AutoModel.from_pretrained(model, subfolder="vae", dtype=_torch_dtype(dtype)).to(device)
    down = 2 ** (len(vae.config.block_out_channels) - 1)
    scale = getattr(vae.config, "scaling_factor", None) or 1.0
    shift = getattr(vae.config, "shift_factor", None) or 0.0

    @torch.no_grad()
    def decode(seed: int, width: int, height: int) -> Image.Image:
        # Unit-Gaussian latents in the scaled space the diffusion model works in (its starting noise), unscaled
        # exactly as the pipeline does before vae.decode, so the decoder sees latents of the usual size.
        g = torch.Generator("cpu").manual_seed(seed)
        n = torch.randn((1, vae.config.latent_channels, height // down, width // down), generator=g)
        y = vae.decode((n / scale + shift).to(device, vae.dtype)).sample[0].float()
        return _signed_to_pil(y.permute(1, 2, 0).cpu().numpy(), model)

    return decode


def _default_latent_factory(model: str, dtype: str, device: str):
    import numpy as np
    import torch
    from diffusers import AutoModel

    vae = AutoModel.from_pretrained(model, subfolder="vae", dtype=_torch_dtype(dtype)).to(device)
    scale = getattr(vae.config, "scaling_factor", None) or 1.0
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    flat_latents = {}                                   # (size, colour) -> the encoded flat image, in scaled space

    @torch.no_grad()
    def decode(img: Image.Image, seed: int, latent_noise: float) -> Image.Image:
        key = (img.size, img.getpixel((0, 0)))
        if key not in flat_latents:
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
            flat_latents[key] = ((vae.encode(x.to(device, vae.dtype)).latent_dist.mode().float() - shift) * scale).cpu()
        z = flat_latents[key]
        n = torch.randn(z.shape, generator=torch.Generator("cpu").manual_seed(seed))
        y = vae.decode(((z + latent_noise * n) / scale + shift).to(device, vae.dtype)).sample[0].float()
        return _signed_to_pil(y.permute(1, 2, 0).cpu().numpy(), model)

    return decode


def _signed_to_pil(arr, what: str) -> Image.Image:
    """[-1, 1] floats -> uint8 image, refusing NaN/inf (clamp would hide them as black)."""
    import numpy as np
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite pixels from the VAE for {what} (fp16 overflow?)")
    return Image.fromarray(((np.clip(arr, -1, 1) + 1.0) * 127.5).round().astype("uint8"))


class VaeRoundtripBackend:
    """The model's VAE alone, no diffusion (the spec §8 fallback). By default a flat colour image is encoded and
    decoded. Options: `noise_std` adds seeded Gaussian pixel noise to the flat image before encoding;
    `latent = "random"` skips the image and decodes seeded unit-Gaussian latents instead; `latent = "flat+noise"`
    encodes the flat colour once and decodes its latent plus seeded Gaussian noise of std `latent_noise` (in the
    scaled latent space), so all the texture comes from the decoder side."""

    name = "vae-roundtrip"

    def __init__(self, model: str, options: dict, vae_factory=None, decode_factory=None, latent_factory=None):
        self.model = model
        self.device = options.get("device") or pick_device()
        self.dtype = pick_dtype(self.device, options.get("dtype", "auto"))
        self.noise_std = float(options.get("noise_std", 0.0))
        self.latent = options.get("latent", "encode")
        if self.latent not in ("encode", "random", "flat+noise"):
            raise ValueError(f"vae-roundtrip option latent must be 'encode', 'random' or 'flat+noise', got {self.latent!r}")
        self.latent_noise = float(options.get("latent_noise", 0.0))
        if self.latent == "flat+noise" and self.latent_noise <= 0:
            raise ValueError("vae-roundtrip latent = 'flat+noise' needs latent_noise > 0")
        self._factory = vae_factory or (lambda m, d: _default_vae_factory(m, d, self.device))
        self._decode_factory = decode_factory or (lambda m, d: _default_decode_factory(m, d, self.device))
        self._latent_factory = latent_factory or (lambda m, d: _default_latent_factory(m, d, self.device))
        self._fn = None
        self._prov = provenance(self.device)

    def generate(self, job: GenJob) -> GenOutput:
        extra = {"dtype": self.dtype, "vae_input": self.latent, **self._prov}
        if self.latent == "random":
            if self._fn is None:
                self._fn = self._decode_factory(self.model, self.dtype)
            out = self._fn(job.seed, job.width, job.height)
            return GenOutput(encode_png(out), "png", self.device, extra)
        if self._fn is None:
            self._fn = (self._latent_factory if self.latent == "flat+noise" else self._factory)(self.model, self.dtype)
        colour = (FLAT_RGB.get(job.prompt) or FLAT_RGB.get(COLOUR_OF_PROMPT.get(job.prompt, ""))
                  or rgb_of_prompt(job.prompt))
        if colour is None:
            raise ValueError(f"vae-roundtrip prompt must be one of {sorted(FLAT_RGB)} or carry RGB(r, g, b), "
                             f"got '{job.prompt}'")
        img = Image.new("RGB", (job.width, job.height), colour)
        if self.latent == "flat+noise":
            extra["latent_noise"] = self.latent_noise
            return GenOutput(encode_png(self._fn(img, job.seed, self.latent_noise)), "png", self.device, extra)
        if self.noise_std:
            import numpy as np
            noise = np.random.default_rng(job.seed).normal(0.0, self.noise_std, (job.height, job.width, 3))
            img = Image.fromarray(np.clip(np.rint(np.asarray(img, np.float64) + noise), 0, 255).astype(np.uint8))
            extra["noise_std"] = self.noise_std
        out = self._fn(img)
        return GenOutput(encode_png(out), "png", self.device, extra)
