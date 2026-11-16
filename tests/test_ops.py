import io

import pytest
from PIL import Image

from flatprint.ops import apply_ops, parse_ops


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def test_parse_ops():
    assert parse_ops("crop:512,resize:0.5,jpeg:85") == [("crop", 512), ("resize", 0.5), ("jpeg", 85)]
    assert parse_ops("png") == [("png", None)]


@pytest.mark.parametrize("bad", ["jpeg:85,crop:10", "crop:10", "png,jpeg:90", "blur:2,png", "jpeg:150"])
def test_parse_ops_rejects(bad):
    with pytest.raises(ValueError):
        parse_ops(bad)


def test_crop_is_centre_and_lossless(tiny_rgb):
    img = Image.new("RGB", (10, 6))
    img.putpixel((5, 3), (255, 0, 0))
    out = Image.open(io.BytesIO(apply_ops(png_bytes(img), parse_ops("crop:4,png"))))
    assert out.size == (4, 4) and out.getpixel((2, 2)) == (255, 0, 0)


def test_crop_larger_than_image_raises(tiny_rgb):
    with pytest.raises(ValueError, match="smaller than crop"):
        apply_ops(png_bytes(tiny_rgb(8, 8)), parse_ops("crop:16,png"))


def test_jpeg_and_resize(tiny_rgb):
    out = apply_ops(png_bytes(tiny_rgb(64, 32)), parse_ops("resize:0.5,jpeg:70"))
    assert out[:3] == b"\xff\xd8\xff"
    assert Image.open(io.BytesIO(out)).size == (32, 16)


def test_deterministic(tiny_rgb):
    data = png_bytes(tiny_rgb(64, 64, (10, 200, 30)))
    ops = parse_ops("resize:0.75,jpeg:80")
    assert apply_ops(data, ops) == apply_ops(data, ops)


def test_maxwidth_shrinks_to_the_width_keeping_the_aspect(tiny_rgb):
    out = Image.open(io.BytesIO(apply_ops(png_bytes(tiny_rgb(1216, 832)), parse_ops("maxwidth:736,png"))))
    assert out.size == (736, round(832 * 736 / 1216))


def test_maxwidth_never_enlarges(tiny_rgb):
    out = Image.open(io.BytesIO(apply_ops(png_bytes(tiny_rgb(512, 512)), parse_ops("maxwidth:736,png"))))
    assert out.size == (512, 512)


def test_maxwidth_needs_a_positive_width():
    with pytest.raises(ValueError):
        parse_ops("maxwidth:0,png")
