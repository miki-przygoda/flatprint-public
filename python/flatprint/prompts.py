"""Prompt sets: flat (blank) colours and captions for content images."""

import json
import random
import re
from pathlib import Path

FLAT_COLOURS = {
    "black": "a plain solid black image, uniform flat black, no objects, no texture",
    "white": "a plain solid white image, uniform flat white, no objects, no texture",
    "grey": "a plain solid mid-grey image, uniform flat grey, no objects, no texture",
}


# The exact value goes in the prompt (decimal and hex) so the record carries the intended colour.
# The model is only asked for it: measure the real pixels, don't assume them.
RGB_TEMPLATE = ("a plain solid flat image filled with one exact colour, RGB({r}, {g}, {b}), hex #{r:02x}{g:02x}{b:02x}, "
                "{name}, uniform, no objects, no texture, no gradient")
_RGB_IN_PROMPT = re.compile(r"RGB\((\d{1,3}), (\d{1,3}), (\d{1,3})\)")


def _rgb_prompt(entry: dict) -> str:
    rgb = entry.get("rgb")
    if not (isinstance(rgb, list) and len(rgb) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in rgb)):
        raise ValueError(f"colour {entry.get('name')!r}: rgb must be three integers 0-255, got {rgb}")
    return RGB_TEMPLATE.format(r=rgb[0], g=rgb[1], b=rgb[2], name=entry.get("name") or "flat colour")

FLAT_RGB = {"black": (0, 0, 0), "white": (255, 255, 255), "grey": (128, 128, 128)}


def flat_prompts(colours: list) -> list[str]:
    """Each colour is a known name ("black") or {name, rgb} for an exact value."""
    unknown = [c for c in colours if not isinstance(c, dict) and c not in FLAT_COLOURS]
    if unknown:
        raise ValueError(f"unknown flat colour(s) {unknown}; known: {sorted(FLAT_COLOURS)} or {{name, rgb}}")
    return [_rgb_prompt(c) if isinstance(c, dict) else FLAT_COLOURS[c] for c in colours]


def rgb_of_prompt(prompt: str) -> tuple[int, int, int] | None:
    m = _RGB_IN_PROMPT.search(prompt)
    return tuple(int(v) for v in m.groups()) if m else None


def _read_captions(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        caps = [a["caption"] for a in data["annotations"]]
    else:
        caps = path.read_text(encoding="utf-8").splitlines()
    seen, out = set(), []
    for c in (c.strip() for c in caps):
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def caption_prompts(path: Path, n: int, seed: int, exclude: list[str] | None = None) -> list[str]:
    """n captions sampled with `seed`. Captions holding any `exclude` word (whole word, any case: real people,
    brands, named organisations) are dropped first, so the sample stays deterministic for one exclude list."""
    caps = _read_captions(Path(path))
    order = random.Random(seed).sample(caps, len(caps))   # the whole pool in one seeded order, then filtered:
    if exclude:                                           # excluding a word only replaces the captions holding it
        bad = re.compile(r"\b(" + "|".join(re.escape(w) for w in exclude) + r")\b", re.IGNORECASE)
        order = [c for c in order if not bad.search(c)]
    if len(order) < n:
        raise ValueError(f"asked for {n} captions but {path} has only {len(order)} unique")
    return order[:n]
