import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
import cleanup_lib as cl  # noqa: E402


def _flat(h, w, v=10):
    return np.full((h, w, 3), v, np.uint8)


def test_thumbnail_of_a_flat_image_is_flat_at_any_size():
    for h, w in [(512, 512), (768, 768), (1280, 1280), (100, 100)]:
        t = cl.thumbnail(_flat(h, w, 37), n=32)
        assert t.shape == (32, 32, 3) and t.dtype == np.float32 and np.allclose(t, 37, atol=1e-3)


def test_thumbnail_is_the_block_mean():
    px = np.zeros((64, 64, 3), np.uint8)
    px[:8, :8] = 80                                     # one 8x8 block bright, then a quarter of the next one
    px[:4, 8:12] = 80
    t = cl.thumbnail(px, n=8)
    assert t[0, 0, 0] == pytest.approx(80) and t[0, 1, 0] == pytest.approx(20) and t[1, 1, 0] == 0


def test_identical_thumbnails_score_zero():
    t = cl.thumbnail(_flat(64, 64), n=16)
    assert cl.scores(t, t) == pytest.approx({"brightness": 0, "shape": 0, "spot": 0})


def test_a_uniform_offset_is_brightness_only():
    ref = cl.thumbnail(_flat(64, 64, 10), n=16)
    s = cl.scores(cl.thumbnail(_flat(64, 64, 60), n=16), ref)
    assert s["brightness"] == pytest.approx(50) and s["shape"] == pytest.approx(0) and s["spot"] == pytest.approx(0)


def test_a_white_frame_scores_on_shape():
    px = np.full((128, 128, 3), 255, np.uint8)
    px[16:-16, 16:-16] = 0
    s = cl.scores(cl.thumbnail(px, n=32), cl.thumbnail(_flat(128, 128, 0), n=32))
    assert s["shape"] > 50 and s["spot"] > 100


def test_a_small_bright_mark_scores_on_spot_more_than_shape():
    px = _flat(128, 128, 0)
    px[60:68, 60:68] = 255                              # e.g. a bit of text
    s = cl.scores(cl.thumbnail(px, n=32), cl.thumbnail(_flat(128, 128, 0), n=32))
    assert s["spot"] > 200 and s["shape"] < 0.3 * s["spot"]


def test_calibrate_scores_each_reference_against_the_others():
    thumbs = np.stack([cl.thumbnail(_flat(32, 32, v), n=8) for v in (10, 10, 10, 40)])
    out = cl.calibrate(thumbs)
    assert out.shape == (4, 3)
    assert out[3, 0] == pytest.approx(30)              # 40 against the mean of three 10s
    assert out[0, 0] == pytest.approx(10)              # 10 against the mean of 10, 10, 40 = 20
