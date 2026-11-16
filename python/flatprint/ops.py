"""Deterministic image op chains: centre crop, resize, max width, and one final encoder."""

import io

from PIL import Image

ENCODERS = {"jpeg", "png", "webp"}
TRANSFORMS = {"crop", "resize", "maxwidth"}


def parse_ops(spec: str) -> list[tuple[str, float | int | None]]:
    ops = []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        name, _, arg = item.partition(":")
        if name not in ENCODERS | TRANSFORMS:
            raise ValueError(f"unknown op '{name}' in '{spec}'")
        if name == "png":
            ops.append(("png", None))
        elif name == "resize":
            f = float(arg)
            if not 0 < f <= 4:
                raise ValueError(f"resize factor must be in (0, 4], got {f}")
            ops.append(("resize", f))
        else:
            n = int(arg)
            if name in {"jpeg", "webp"} and not 1 <= n <= 100:
                raise ValueError(f"{name} quality must be 1..100, got {n}")
            if name == "maxwidth" and n < 1:
                raise ValueError(f"maxwidth must be at least 1, got {n}")
            ops.append((name, n))
    enc = [i for i, (n, _) in enumerate(ops) if n in ENCODERS]
    if len(enc) != 1 or enc[0] != len(ops) - 1:
        raise ValueError(f"'{spec}' must end with exactly one encoder (jpeg:Q, png or webp:Q)")
    return ops


def format_ops(ops) -> str:
    """Canonical spelling of a parsed op chain (used for ids and provenance)."""
    return ",".join(name if arg is None else f"{name}:{arg}" for name, arg in ops)


def apply_ops(data: bytes, ops) -> bytes:
    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGB")
    out = io.BytesIO()
    for name, arg in ops:
        if name == "crop":
            w, h = img.size
            if w < arg or h < arg:
                raise ValueError(f"image {w}x{h} is smaller than crop {arg}")
            left, top = (w - arg) // 2, (h - arg) // 2
            img = img.crop((left, top, left + arg, top + arg))
        elif name == "resize":
            img = img.resize((max(1, round(img.width * arg)), max(1, round(img.height * arg))),
                             Image.Resampling.BICUBIC)
        elif name == "maxwidth":            # shrink to at most arg px wide, like a site's display copy; never enlarge
            if img.width > arg:
                img = img.resize((arg, max(1, round(img.height * arg / img.width))), Image.Resampling.BICUBIC)
        elif name in ENCODERS:
            img = Image.frombytes(img.mode, img.size, img.tobytes())  # drop ICC/EXIF/text from the source
            _encode(img, name, arg, out)
    return out.getvalue()


def _encode(img: Image.Image, name: str, arg, out: io.BytesIO) -> None:
    if name == "jpeg":
        img.save(out, "JPEG", quality=arg, subsampling="4:2:0", optimize=False)
    elif name == "webp":
        img.save(out, "WEBP", quality=arg, method=4)
    else:
        img.save(out, "PNG")
