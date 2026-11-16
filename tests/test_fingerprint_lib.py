import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
import fingerprint_lib as fl  # noqa: E402

RNG = np.random.default_rng(0)


def _stack(n=8, h=32, w=32, base=100, noise=2.0, pattern=None):
    x = base + RNG.normal(0, noise, (n, h, w, 3))
    if pattern is not None:
        x = x + pattern
    return np.clip(x, 0, 255).astype(np.uint8)


def _checker(h=32, w=32, amp=3.0):
    yy, xx = np.mgrid[0:h, 0:w]
    return (amp * np.where((xx // 4 + yy // 4) % 2, 1, -1))[..., None].repeat(3, axis=2)


def test_mean_residual_has_zero_mean_per_image_and_channel():
    r = fl.residuals(_stack(), "mean")
    assert r.dtype == np.float32 and np.allclose(r.mean(axis=(1, 2)), 0, atol=1e-4)


def test_target_residual_subtracts_the_target_colour():
    px = np.full((1, 4, 4, 3), 70, np.uint8)
    assert np.allclose(fl.residuals(px, "target", target=(64, 64, 64)), 6)


def test_highpass_removes_flat_and_smooth_parts_but_keeps_fine_pattern():
    h = w = 64
    ramp = np.linspace(0, 40, w)[None, :, None].repeat(h, 0).repeat(3, 2)
    smooth = np.clip(100 + ramp, 0, 255).astype(np.uint8)[None]
    assert np.abs(fl.residuals(smooth, "highpass", sigma=3)[0, 8:-8, 8:-8]).max() < 1.0
    fine = np.clip(100 + _checker(h, w, 10), 0, 255).astype(np.uint8)[None]
    assert np.abs(fl.residuals(fine, "highpass", sigma=3)).mean() > 5


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        fl.residuals(_stack(), "median")


def test_fingerprint_sum_in_chunks_matches_direct_mean():
    px = _stack(n=10)
    total, n = fl.fingerprint_sum(px, list(range(10)), "mean", chunk=3)
    assert n == 10 and np.allclose(total / n, fl.residuals(px, "mean").mean(axis=0), atol=1e-4)


def test_leave_one_out_drops_that_image():
    r = fl.residuals(_stack(n=5), "mean").astype(np.float64)
    assert np.allclose(fl.leave_one_out(r.sum(0), 5, r[2]), np.delete(r, 2, 0).mean(0))


def test_corr_of_identical_is_one_and_of_opposite_is_minus_one():
    a = RNG.normal(size=(8, 8, 3))
    assert fl.corr(a, a) == pytest.approx(1.0) and fl.corr(a, -a) == pytest.approx(-1.0)


def test_split_half_curve_rises_with_n_when_there_is_a_shared_pattern():
    res = fl.residuals(_stack(n=32, noise=6.0, pattern=_checker(amp=2.0)), "mean")
    curve = fl.split_half_curve(res, sizes=[1, 4, 16], repeats=5, rng=np.random.default_rng(1))
    means = [np.mean(curve[n]) for n in (1, 4, 16)]
    assert means[0] < means[1] < means[2] and means[2] > 0.5


def test_split_half_curve_is_near_zero_for_pure_noise():
    res = fl.residuals(_stack(n=32, noise=6.0), "mean")
    curve = fl.split_half_curve(res, sizes=[16], repeats=5, rng=np.random.default_rng(1))
    assert abs(np.mean(curve[16])) < 0.1


def test_find_peaks_locates_a_period_8_pattern():
    x = np.arange(256)
    img = np.repeat((128 + 10 * np.cos(2 * np.pi * x / 8))[None, :, None], 256, 0).repeat(3, 2)
    rows = fl.find_peaks(fl.log_spectrum(img, hann=True), n=1, radius=4, skip_axes=False, rank_by="contrast")
    assert (rows[0]["fx"], rows[0]["fy"], rows[0]["period x (px)"]) == (32, 0, 8.0)


def test_lattice_is_one_of_each_mirrored_pair_without_the_centre():
    pts = fl.lattice(32, 32, div=16)                       # steps of 2 bins, fx, fy in [-16, 15]
    assert (0, 0) not in pts and (2, 0) in pts and (-2, 0) not in pts and (-2, 2) in pts
    assert all(fx % 2 == 0 and fy % 2 == 0 for fx, fy in pts)
    assert len(pts) == len(set(pts))


def test_lattice_strength_reads_the_contrast_at_each_point():
    spec = np.zeros((64, 64))
    spec[32 + 4, 32 + 8] = 10.0                             # (fx=8, fy=4)
    s = fl.lattice_strength(spec, [(8, 4), (0, 4)])
    assert s[0] > 9 and abs(s[1]) < 0.1


def test_guides_draw_lines_at_eighths():
    out = fl.with_guides(np.zeros((64, 64), np.uint8), divs=(8,))
    assert out.shape == (64, 64, 3)
    assert out[10, 32 + 8].any() and not out[10, 32 + 5].any()


def test_zoom_blows_up_a_crop_with_square_pixels():
    img = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    z = fl.zoom(img, x=10, y=20, size=4, scale=5)
    assert z.shape == (20, 20, 3) and (z[:5, :5] == img[20, 10]).all()


def test_fold_returns_the_repeating_tile():
    tile = RNG.normal(size=(16, 16, 3))
    img = np.tile(tile, (4, 6, 1))                          # 64 x 96
    assert np.allclose(fl.fold(img, 16), tile)


def test_fold_of_a_batch_folds_each_image():
    tiles = RNG.normal(size=(3, 8, 8, 3))
    batch = np.stack([np.tile(t, (4, 4, 1)) for t in tiles])
    out = fl.fold(batch, 8)
    assert out.shape == (3, 8, 8, 3) and np.allclose(out, tiles)


def test_fold_averages_away_what_does_not_repeat():
    noise = RNG.normal(size=(512, 512, 3))
    assert fl.fold(noise, 16).std() < 0.1 * noise.std()


def test_fold_needs_a_whole_number_of_tiles():
    with pytest.raises(ValueError, match="multiple"):
        fl.fold(np.zeros((100, 96, 3)), 16)


def test_percentile_of_places_a_value_among_others():
    assert fl.percentile_of(5, [1, 2, 3, 4, 6, 7, 8, 9]) == pytest.approx(50)
    assert fl.percentile_of(10, [1, 2, 3]) == pytest.approx(100)


def test_shift_corr_is_one_at_the_shift_that_lines_tiles_up():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(16, 16, 3))
    c = fl.shift_corr(a, np.roll(a, (-3, 5), axis=(0, 1)))
    assert c.shape == (16, 16)
    assert c[3, 11] == pytest.approx(1.0)               # rolling b by (3, -5) = (3, 11) mod 16 undoes it
    assert np.argmax(c) == 3 * 16 + 11


def test_flat_noise_is_reproducible_around_the_level():
    a = fl.flat_noise(3, (64, 32), level=8, sigma=2.0, seed=5)
    assert a.shape == (3, 32, 64, 3) and a.dtype == np.uint8
    assert np.array_equal(a, fl.flat_noise(3, (64, 32), level=8, sigma=2.0, seed=5))
    assert abs(a.mean() - 8) < 0.2 and 1.8 < a.astype(float).std() < 2.2


def test_roundtrip_png_is_lossless_and_jpeg_is_not():
    px = fl.flat_noise(1, (32, 32), level=100, sigma=20, seed=0)[0]
    assert np.array_equal(fl.roundtrip(px, "png"), px)
    out = fl.roundtrip(px, "jpeg:50")
    assert out.shape == px.shape and out.dtype == np.uint8 and not np.array_equal(out, px)


def test_masked_fold_with_a_full_mask_is_the_plain_fold():
    rng = np.random.default_rng(3)
    res = rng.normal(size=(32, 48, 3))
    assert np.allclose(fl.masked_fold(res, np.ones((32, 48), bool), 16), fl.fold(res, 16), atol=1e-6)


def test_masked_fold_averages_only_the_chosen_pixels():
    res = np.zeros((32, 32, 3))
    res[:16, :16] = 1.0                                  # top-left tile 1, the other three tiles 0
    mask = np.zeros((32, 32), bool)
    mask[:16, :16] = True
    assert np.allclose(fl.masked_fold(res, mask, 16), 1.0)
    empty = fl.masked_fold(res, np.zeros((32, 32), bool), 16)
    assert np.isfinite(empty).all() and (empty == 0).all()     # nothing chosen: zeros, not NaN


def test_region_masks_split_dark_bright_smooth_and_detailed():
    px = np.zeros((64, 64, 3), np.uint8)
    px[:, 32:] = 200                                     # right half bright
    rng = np.random.default_rng(0)
    px[:32] = np.clip(px[:32].astype(int) + rng.normal(0, 25, (32, 64, 3)), 0, 255).astype(np.uint8)  # top half textured
    m = fl.region_masks(px, frac=0.25, sigma=2)
    assert set(m) == {"dark", "bright", "smooth", "detailed"}
    for k, v in m.items():
        assert v.shape == (64, 64) and v.dtype == bool and 0.2 < v.mean() < 0.3, k
    assert m["dark"][:, 32:].sum() == 0 and m["bright"][:, :32].sum() == 0
    assert m["smooth"][32:].sum() > m["smooth"][:32].sum() and m["detailed"][:32].sum() > m["detailed"][32:].sum()


def test_slope_is_how_much_of_the_signature_an_image_holds():
    rng = np.random.default_rng(4)
    s = rng.normal(size=(16, 16, 3))
    assert fl.slope(2.0 * s + 5.0, s) == pytest.approx(2.0)      # an offset doesn't change it
    assert fl.slope(-s, s) == pytest.approx(-1.0)
    noise = rng.normal(size=(200, 16, 16, 3))
    assert abs(np.mean([fl.slope(n, s) for n in noise])) < 0.05  # unrelated folds: about 0 on average
    assert fl.slope(s, np.zeros_like(s)) == 0.0                  # a flat signature holds no pattern


def _two_blobs(n_pos=300, n_neg=30, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([2.0, 1.0], 0.5, (n_pos, 2)), rng.normal([0.0, 0.0], 0.5, (n_neg, 2))])
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    return X, y


def test_logistic_ranks_the_two_classes_apart():
    X, y = _two_blobs()
    p = fl.predict_logistic(fl.fit_logistic(X, y), X)
    assert p.shape == (len(y),) and ((0 <= p) & (p <= 1)).all()
    assert p[y == 1].mean() > 0.9 and p[y == 0].mean() < 0.1


def test_balanced_logistic_ignores_the_class_counts():
    X, y = _two_blobs(n_pos=1000, n_neg=20)
    mid = np.array([[1.0, 0.5]])                              # halfway between the two class centres
    balanced = fl.predict_logistic(fl.fit_logistic(X, y, balanced=True), mid)[0]
    plain = fl.predict_logistic(fl.fit_logistic(X, y, balanced=False), mid)[0]
    assert abs(balanced - 0.5) < 0.15 and plain > balanced    # unweighted, the 50:1 majority pulls it up


def test_cross_validated_probabilities_come_from_models_that_did_not_see_them():
    X, y = _two_blobs(seed=1)
    p, folds = fl.cv_logistic(X, y, k=5, seed=3)
    assert p.shape == (len(y),) and set(folds) == set(range(5))
    f0 = folds == 0
    manual = fl.predict_logistic(fl.fit_logistic(X[~f0], y[~f0]), X[f0])
    assert np.allclose(p[f0], manual)
    for c in (0, 1):                                          # stratified: each class spread over every fold
        assert len(set(folds[y == c])) == 5


def _tiled(size=1024, seed=0, amp=6.0):
    """A grey picture carrying a fixed 16-px tile (the pattern) plus fine noise; returns (pixels, tile)."""
    rng = np.random.default_rng(seed)
    tile = rng.normal(0, amp, (16, 16, 3))
    img = 128 + np.tile(tile, (size // 16, size // 16, 1)) + rng.normal(0, 1.0, (size, size, 3))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), tile


def test_scale_scan_finds_the_resize_and_recovers_the_pattern():
    from PIL import Image
    px, tile = _tiled()
    small = np.asarray(Image.fromarray(px).resize((614, 614), Image.Resampling.BICUBIC))   # x0.6, like a site
    scan = fl.scale_scan(small, np.round(np.arange(0.50, 1.001, 0.05), 2), {"ref": tile}, tile)
    best = max(scan, key=lambda r: r["strength"])
    at_one = next(r for r in scan if r["scale"] == 1.0)
    assert abs(best["scale"] - 0.6) <= 0.05
    assert best["strength"] > 0.1 and best["strength"] > 50 * abs(at_one["strength"])   # shrinking blurs part of it away
    assert {"scale", "strength", "score"} <= set(best)


def test_scale_scan_skips_scales_that_would_make_the_picture_too_big():
    px, tile = _tiled(size=512)
    scan = fl.scale_scan(px, [0.4, 0.5, 0.8, 1.0], {"ref": tile}, tile, max_width=1000)
    assert [r["scale"] for r in scan] == [0.8, 1.0]       # 512/0.4 = 1280 and 512/0.5 = 1024 are over 1000
    assert fl.scale_scan(px, [0.4], {"ref": tile}, tile, max_width=600)[0]["scale"] == 1.0   # 1.0 always kept


def test_candidate_scales_are_exact_factors_from_original_widths():
    sc = fl.candidate_scales(736, max_width=1216, step=16)
    assert sc[0] == 1.0 and len(sc) == 1 + len(range(752, 1217, 16))
    assert any(abs(x - 736 / 1216) < 1e-12 for x in sc)       # the exact Pinterest factor for a 1216-wide original
    assert fl.candidate_scales(4032, max_width=2048) == [1.0]  # wider than any original: only as it is


def test_candidate_scales_can_come_from_a_list_of_usual_widths():
    sc = fl.candidate_scales(736, widths=range(512, 2049, 64))
    assert sc[0] == 1.0 and sc[1:] == [736 / w for w in range(768, 2049, 64)]
    assert fl.candidate_scales(2100, widths=range(512, 2049, 64)) == [1.0]


def test_best_of_k_null_grows_with_k_and_matches_the_pool_at_k1():
    pool = np.random.default_rng(0).normal(0.0, 1.0, 5000)
    m1, s1 = fl.best_of_k_null(pool, 1, draws=20000, seed=1)
    m25, s25 = fl.best_of_k_null(pool, 25, draws=20000, seed=1)
    assert abs(m1) < 0.05 and abs(s1 - 1.0) < 0.05
    assert 1.8 < m25 < 2.2 and s25 < s1                 # the best of 25 normals: mean ~1.97, narrower spread


def test_peak_sharpness_is_high_for_one_spike_and_low_for_noise():
    rng = np.random.default_rng(5)
    flat = rng.normal(0, 0.01, 20)
    spiked = flat.copy()
    spiked[7] = 0.2
    assert fl.peak_sharpness(spiked) > 10 > fl.peak_sharpness(flat) > 0
    assert np.isnan(fl.peak_sharpness([0.1, 0.2, 0.3]))        # too few scales for a curve


def test_matched_z_compares_with_the_nearest_reference_group_and_leaves_itself_out():
    refs = [(10, v) for v in (0.0, 0.1, 0.2, 0.3)] + [(30, v) for v in (1.0, 1.1, 1.2, 1.3)]
    z = fl.matched_z(0.15, 11, refs, tolerance=3)                 # compared with the k=10 group only
    assert abs(z - (0.15 - 0.15) / np.std([0.0, 0.1, 0.2, 0.3], ddof=1)) < 1e-9
    z_self = fl.matched_z(0.0, 10, refs, tolerance=3, leave_out=(10, 0.0))
    assert abs(z_self - (0.0 - 0.2) / np.std([0.1, 0.2, 0.3], ddof=1)) < 1e-9
    assert np.isnan(fl.matched_z(0.5, 20, refs, tolerance=3))     # nobody within 3 scales


def test_grouped_cv_never_trains_on_the_lineage_it_predicts():
    X, y = _two_blobs(n_pos=120, n_neg=40, seed=2)
    groups = np.r_[np.arange(120) // 4, 1000 + np.arange(40) // 2]        # positives in lineages of 4, negatives of 2
    train = np.ones(len(y), bool)
    train[::7] = False                                                    # some rows are only predicted (e.g. copies)
    p, folds = fl.grouped_cv_logistic(X, y, train, groups, k=5, seed=0)
    assert p.shape == (len(y),) and np.isfinite(p).all()
    for g in np.unique(groups):                                           # one fold per lineage
        assert len(set(folds[groups == g])) == 1
    f0 = folds == 0                                                       # = a model fitted without fold 0's lineages
    manual = fl.predict_logistic(fl.fit_logistic(X[train & ~f0], y[train & ~f0]), X[f0])
    assert np.allclose(p[f0], manual)


def test_peak_sharpness_of_a_flat_curve_is_zero_not_infinite():
    assert fl.peak_sharpness([0.0] * 10) == 0.0               # e.g. an all-black picture: every scale scores 0
    assert fl.peak_sharpness([0.0] * 9 + [0.3]) == float("inf")   # a real spike over a perfectly flat rest
