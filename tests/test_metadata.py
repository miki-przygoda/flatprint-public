import io
import struct
import zlib

from PIL import Image, PngImagePlugin

from flatprint.metadata import inspect_bytes


def png_with_text(tiny_rgb):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "steps: 30, sampler: euler")
    b = io.BytesIO()
    tiny_rgb().save(b, "PNG", pnginfo=info)
    return b.getvalue()


def add_png_chunk(data: bytes, ctype: bytes, payload: bytes) -> bytes:
    chunk = struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", zlib.crc32(ctype + payload))
    iend = data.rindex(b"IEND") - 4
    return data[:iend] + chunk + data[iend:]


def test_png_text_and_chunks(tiny_rgb):
    r = inspect_bytes(png_with_text(tiny_rgb))
    assert r["format"] == "png" and "parameters" in r["text_keys"]
    assert r["png_chunks"][0] == "IHDR" and r["png_chunks"][-1] == "IEND"
    assert r["c2pa"] is False


def test_png_c2pa_cabx_chunk(tiny_rgb):
    data = add_png_chunk(png_with_text(tiny_rgb), b"caBX", b"jumb....c2pa....")
    assert inspect_bytes(data)["c2pa"] is True


def test_jpeg_exif_and_c2pa_app11(tiny_rgb):
    img = tiny_rgb()
    exif = Image.Exif()
    exif[0x0131] = "SomeGenerator 1.0"        # Software tag
    b = io.BytesIO()
    img.save(b, "JPEG", exif=exif.tobytes())
    data = b.getvalue()
    app11 = b"\xff\xeb" + struct.pack(">H", 2 + 16) + b"JP\x00\x01jumbc2pa\x00\x00\x00\x00"
    data = data[:2] + app11 + data[2:]
    r = inspect_bytes(data)
    assert r["format"] == "jpg" and r["exif"] is True and r["c2pa"] is True
    assert "APP11" in r["jpeg_segments"] and "APP1" in r["jpeg_segments"]


def test_clean_png_has_nothing(tiny_rgb):
    b = io.BytesIO()
    tiny_rgb().save(b, "PNG")
    r = inspect_bytes(b.getvalue())
    assert (r["text_keys"], r["exif"], r["xmp"], r["c2pa"]) == ([], False, False, False)
