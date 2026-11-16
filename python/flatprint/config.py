"""Run configs (TOML) and their expansion into generation jobs."""

import hashlib
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from flatprint.prompts import caption_prompts, flat_prompts

BACKENDS = {"diffusers", "vae-roundtrip", "openai", "gemini"}
DTYPES = {"auto", "float32", "float16", "bfloat16"}
OFFLOAD = {"none", "model", "sequential"}
TOP_KEYS = {"name", "set", "backend", "model", "steps", "guidance", "sizes", "seeds", "prompts", "options"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GenJob:
    set: str
    model: str
    backend: str
    prompt: str
    seed: int
    width: int
    height: int
    steps: int | None
    guidance: float | None

    def key(self) -> str:
        raw = "|".join(str(v) for v in (self.set, self.model, self.backend, self.prompt, self.seed,
                                         self.width, self.height, self.steps, self.guidance))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RunConfig:
    name: str
    set: str
    backend: str
    model: str
    prompts: list[str]
    sizes: list[tuple[int, int]]
    seed_start: int
    seeds_per_prompt: int
    seed_pick: int | None = None   # set: seeds are drawn at random (reproducibly) instead of counting up
    steps: int | None = None
    guidance: float | None = None
    options: dict = field(default_factory=dict)


def _resolve(path: Path, file: str) -> Path:
    """Relative caption paths resolve against the config's folder, then the cwd."""
    f = Path(file)
    if f.is_absolute():
        return f
    return path.parent / f if (path.parent / f).exists() else f


def load_config(path: Path) -> RunConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = set(raw) - TOP_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(TOP_KEYS)}")
    for k in ("name", "set", "backend", "model", "sizes", "seeds", "prompts"):
        if k not in raw:
            raise ConfigError(f"{path}: missing required key '{k}'")
    if raw["backend"] not in BACKENDS:
        raise ConfigError(f"{path}: unknown backend '{raw['backend']}'; known: {sorted(BACKENDS)}")

    opts = raw.get("options", {})
    if opts.get("dtype", "auto") not in DTYPES:
        raise ConfigError(f"{path}: options.dtype must be one of {sorted(DTYPES)}, got '{opts['dtype']}'")
    if opts.get("offload", "none") not in OFFLOAD:
        raise ConfigError(f"{path}: options.offload must be one of {sorted(OFFLOAD)}, got '{opts['offload']}'")

    p = raw["prompts"]
    if raw["backend"] == "vae-roundtrip" and p.get("kind") != "flat":
        raise ConfigError(f"{path}: backend vae-roundtrip needs prompts.kind = 'flat'")
    try:
        if p.get("kind") == "flat":
            prompts = flat_prompts(p["colours"])
        elif p.get("kind") == "captions":
            f = _resolve(path, p["file"])
            if not f.exists():
                raise ConfigError(f"{path}: captions file '{p['file']}' not found")
            prompts = caption_prompts(f, n=p["n"], seed=p.get("seed", 0), exclude=p.get("exclude"))
        else:
            raise ConfigError(f"{path}: prompts.kind must be 'flat' or 'captions'")
    except ConfigError:
        raise
    except (KeyError, ValueError) as e:
        raise ConfigError(f"{path}: bad [prompts] section: {e}") from e

    seeds = raw["seeds"]
    if seeds.get("random") and "pick_seed" not in seeds:
        raise ConfigError(f"{path}: seeds.random needs seeds.pick_seed so the draw can be repeated")

    return RunConfig(
        name=raw["name"], set=raw["set"], backend=raw["backend"], model=raw["model"],
        prompts=prompts, sizes=[tuple(s) for s in raw["sizes"]],
        seed_start=seeds.get("start", 0), seeds_per_prompt=seeds.get("per_prompt", 1),
        seed_pick=seeds["pick_seed"] if seeds.get("random") else None,
        steps=raw.get("steps"), guidance=raw.get("guidance"), options=raw.get("options", {}),
    )


def expand_jobs(cfg: RunConfig) -> list[GenJob]:
    jobs = []
    n = len(cfg.prompts) * cfg.seeds_per_prompt
    drawn = random.Random(cfg.seed_pick).sample(range(2**31), n) if cfg.seed_pick is not None else None
    for i, prompt in enumerate(cfg.prompts):
        for k in range(cfg.seeds_per_prompt):
            j = i * cfg.seeds_per_prompt + k
            seed = drawn[j] if drawn else cfg.seed_start + j
            for w, h in cfg.sizes:
                jobs.append(GenJob(cfg.set, cfg.model, cfg.backend, prompt, seed, w, h, cfg.steps, cfg.guidance))
    return jobs
