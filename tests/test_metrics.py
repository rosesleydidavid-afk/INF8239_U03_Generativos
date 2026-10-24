import numpy as np
import pytest

from inf8239_u03_gen.metrics import (
    class_entropy,
    fixed_indices,
    fraction_below,
    mean_pairwise_distance,
    mse,
    mse_per_image,
    nearest_neighbor,
    pair_with_different_labels,
    pca_reconstruction,
    per_class_mse,
    psnr,
)


def images(n=6, seed=0):
    return np.random.default_rng(seed).random((n, 28, 28, 1)).astype("float32")


def test_mse_is_zero_for_identical_images_and_known_for_constant_shift():
    x = images()
    assert mse(x, x) == 0
    assert mse(np.zeros((2, 28, 28, 1), "float32"), np.full((2, 28, 28, 1), 0.5, "float32")) == pytest.approx(0.25)


def test_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Formas distintas"):
        mse_per_image(images(3), images(4))


def test_psnr_matches_formula_and_is_finite_for_perfect_reconstruction():
    zeros = np.zeros((2, 28, 28, 1), "float32")
    half = np.full((2, 28, 28, 1), 0.5, "float32")
    assert psnr(zeros, half) == pytest.approx(10 * np.log10(1 / 0.25))
    assert np.isfinite(psnr(zeros, zeros))


def test_per_class_mse_separates_classes():
    original = np.zeros((4, 28, 28, 1), "float32")
    rebuilt = original.copy()
    rebuilt[2:] = 1.0
    result = per_class_mse(original, rebuilt, np.array([0, 0, 1, 1]), n_classes=3)
    assert result[0] == 0 and result[1] == 1 and np.isnan(result[2])


def test_fixed_indices_are_reproducible_unique_and_in_range():
    a, b = fixed_indices(1000, 16, seed=42), fixed_indices(1000, 16, seed=42)
    assert (a == b).all() and len(set(a.tolist())) == 16 and a.min() >= 0 and a.max() < 1000
    assert not (a == fixed_indices(1000, 16, seed=43)).all()
    with pytest.raises(ValueError):
        fixed_indices(5, 10, seed=0)


def test_pair_has_different_labels():
    labels = np.array([0, 0, 1, 2, 2, 3])
    for seed in range(10):
        first, second = pair_with_different_labels(labels, seed)
        assert labels[first] != labels[second]


def test_nearest_neighbor_finds_exact_copy_with_zero_distance():
    reference = images(50)
    queries = np.concatenate([reference[[3, 17]], images(2, seed=9)])
    distances, indices = nearest_neighbor(queries, reference, chunk=2)
    assert indices[0] == 3 and indices[1] == 17
    assert distances[0] == 0 and distances[1] == 0
    assert (distances[2:] > 0).all()


def test_pairwise_distance_zero_for_identical_and_positive_otherwise():
    x = images(1)
    assert mean_pairwise_distance(np.repeat(x, 5, axis=0)) == pytest.approx(0, abs=1e-3)
    assert mean_pairwise_distance(images(8)) > 1
    assert mean_pairwise_distance(images(1)) == 0


def test_class_entropy_bounds():
    assert class_entropy(np.array([10, 0, 0])) == 0
    assert class_entropy(np.full(10, 5)) == pytest.approx(np.log(10))
    assert class_entropy(np.zeros(10)) == 0


def test_fraction_below():
    assert fraction_below(np.array([1, 2, 3, 4]), 3) == 0.5


def test_pca_reconstruction_shape_range_and_improves_with_components():
    train, test = images(200, seed=1), images(20, seed=2)
    small = pca_reconstruction(train, test, 2, seed=0)
    large = pca_reconstruction(train, test, 16, seed=0)
    assert small.shape == test.shape and small.min() >= 0 and small.max() <= 1
    assert mse(test, large) <= mse(test, small)
