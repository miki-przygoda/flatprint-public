"""Byte-level metadata inspection: PNG chunks, JPEG segments, EXIF, XMP, C2PA markers."""

import struct

from flatprint.backends.base import sniff_ext

XMP_NS = b"http://ns.adobe.com/xap/1.0/"


def _png(data: bytes) -> dict:
    chunks, text_keys, pos, exif = [], [], 8, False
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8].decode("latin-1")
        body = data[pos + 8:pos + 8 + length]
        chunks.append(ctype)
        if ctype in ("tEXt", "zTXt", "iTXt"):
            text_keys.append(body.split(b"\x00", 1)[0].decode("latin-1"))
        if ctype == "eXIf":
            exif = True
        pos += 12 + length
        if ctype == "IEND":
            break
    return {"png_chunks": chunks, "text_keys": text_keys, "jpeg_segments": [], "exif": exif,
            "c2pa": "caBX" in chunks}


def _jpeg(data: bytes) -> dict:
    segs, exif, c2pa, pos = [], False, False, 2
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker in (0xD9, 0xDA):           # EOI or start of scan: no more metadata segments
            break
        (length,) = struct.unpack(">H", data[pos + 2:pos + 4])
        body = data[pos + 4:pos + 2 + length]
        if 0xE0 <= marker <= 0xEF:
            segs.append(f"APP{marker - 0xE0}")
            if marker == 0xE1 and body.startswith(b"Exif\x00\x00"):
                exif = True
            if marker == 0xEB and (b"jumb" in body or b"c2pa" in body):
                c2pa = True
        pos += 2 + length
    return {"png_chunks": [], "text_keys": [], "jpeg_segments": segs, "exif": exif, "c2pa": c2pa}


def inspect_bytes(data: bytes) -> dict:
    fmt = sniff_ext(data)
    if fmt == "png":
        r = _png(data)
    elif fmt == "jpg":
        r = _jpeg(data)
    else:
        r = {"png_chunks": [], "text_keys": [], "jpeg_segments": [], "exif": False, "c2pa": b"c2pa" in data}
    return {"format": fmt, **r, "xmp": XMP_NS in data, "size_bytes": len(data)}
