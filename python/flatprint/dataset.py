"""Load a stored set as numpy arrays, one group per flat colour (prompt) and size."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from flatprint.prompts import FLAT_COLOURS, FLAT_RGB, rgb_of_prompt
from flatprint.store import Record, load_manifest, resolve

_NAME_OF_PROMPT = {sentence: name for name, sentence in FLAT_COLOURS.items()}


@dataclass
class Group:
    name: str                              # "dark grey", or the prompt itself if it isn't a flat colour
    prompt: str
    target_rgb: tuple[int, int, int] | None  # what the prompt asked for; the pixels are what the model gave
    size: tuple[int, int]                  # (width, height)
    seeds: list[int]
    records: list[Record]
    pixels: np.ndarray                     # uint8, shape (n, height, width, 3)


def _name(prompt: str) -> str:
    if prompt in _NAME_OF_PROMPT:
        return _NAME_OF_PROMPT[prompt]
    if rgb_of_prompt(prompt):  # RGB template: "..., hex #404040, dark grey, uniform, ..."
        return prompt.split("hex #", 1)[1].split(", ")[1]
    return prompt


def read_ids(path) -> set[str]:
    """Every store id (16 hex chars) mentioned in a text or markdown file, e.g. a hand-made list of outliers."""
    return set(re.findall(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])", Path(path).read_text(encoding="utf-8")))


def load_groups(data_root, set_name: str, exclude: set[str] = frozenset(),
                only: set[str] | None = None) -> list[Group]:
    """Every image in one set (from the merged manifest), grouped by prompt and size, ordered by seed.
    `exclude` drops those ids; `only` keeps just those ids."""
    recs = [r for r in load_manifest(data_root) if r.set == set_name
            and r.id not in exclude and (only is None or r.id in only)]
    by_key: dict[tuple, list[Record]] = {}
    for r in recs:
        by_key.setdefault((r.prompt or "", r.width, r.height), []).append(r)
    order = list(dict.fromkeys(r.prompt or "" for r in sorted(recs, key=lambda r: r.created_at or "")))
    groups = []
    for (prompt, w, h), rs in sorted(by_key.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2])):
        rs.sort(key=lambda r: (r.seed if r.seed is not None else -1, r.id))
        pixels = np.stack([np.asarray(Image.open(resolve(data_root, r)).convert("RGB")) for r in rs])
        name = _name(prompt)
        groups.append(Group(name, prompt, rgb_of_prompt(prompt) or FLAT_RGB.get(name), (w, h),
                            [r.seed for r in rs], rs, pixels))
    return groups


@dataclass
class Matrix:
    records: list[Record]       # row i of pixels is records[i]; ordered by colour, then seed
    names: list[str]            # colour name per row
    pixels: np.ndarray          # uint8 (n, height, width, 3), memory-mapped from the cache dir


def _ordered(recs: list[Record]) -> list[Record]:
    order = {p: i for i, p in enumerate(dict.fromkeys(
        r.prompt or "" for r in sorted(recs, key=lambda r: r.created_at or "")))}
    return sorted(recs, key=lambda r: (order[r.prompt or ""], r.seed if r.seed is not None else -1, r.id))


def load_matrix(data_root, set_name: str, size: tuple[int, int] | None = None) -> Matrix:
    """One set (one size) as a single (n, h, w, 3) array. The first call decodes the PNGs into
    <cache>/<set>-<w>x<h>.npy; later calls memory-map it, and it rebuilds when the set changes. The cache is
    data/cache/ unless FLATPRINT_CACHE names another dir (e.g. a fast disk when the data is on a slow one)."""
    recs = [r for r in load_manifest(data_root) if r.set == set_name]
    sizes = sorted({(r.width, r.height) for r in recs})
    if size is None:
        if len(sizes) > 1:
            raise ValueError(f"set '{set_name}' mixes sizes {', '.join(f'{w}x{h}' for w, h in sizes)}; pass size=(w, h)")
        size = sizes[0] if sizes else (0, 0)
    recs = _ordered([r for r in recs if (r.width, r.height) == tuple(size)])
    names = [_name(r.prompt or "") for r in recs]

    cache = Path(os.environ.get("FLATPRINT_CACHE") or Path(data_root) / "cache")
    stem = f"{set_name}-{size[0]}x{size[1]}"
    npy, key_file = cache / f"{stem}.npy", cache / f"{stem}.json"
    key = json.dumps([[r.id, r.sha256] for r in recs])
    if not (npy.exists() and key_file.exists() and key_file.read_text(encoding="utf-8") == key):
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / f"{stem}.tmp.npy"
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(recs), size[1], size[0], 3))
        for i, r in enumerate(recs):
            out[i] = np.asarray(Image.open(resolve(data_root, r)).convert("RGB"))
        out.flush()
        del out
        os.replace(tmp, npy)
        key_file.write_text(key, encoding="utf-8")
    return Matrix(recs, names, np.load(npy, mmap_mode="r"))
