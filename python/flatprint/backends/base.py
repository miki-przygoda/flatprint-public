"""Backend protocol and byte-level helpers shared by all generators."""

import io
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from flatprint.config import GenJob


@dataclass
class GenOutput:
    data: bytes
    ext: str
    device: str
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def generate(self, job: GenJob) -> GenOutput: ...


def sniff_ext(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"heim", b"heis", b"mif1", b"msf1"):
        return "heic"
    return "bin"


def encode_png(img: Image.Image) -> bytes:
    """Lossless PNG with no metadata chunks (pixels only)."""
    clean = Image.frombytes(img.mode, img.size, img.tobytes())
    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    return buf.getvalue()
