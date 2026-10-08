import numpy as np
import pytest

from src.cv.shapes import CLASSES, make_dataset, make_shape_image


@pytest.mark.parametrize("shape", CLASSES)
def test_image_shape_and_dtype(shape):
    img = make_shape_image(shape, size=64, rng=np.random.default_rng(0))
    assert img.shape == (64, 64)
    assert img.dtype == np.uint8
    assert img.max() > 100  # the shape was actually drawn


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        make_shape_image("hexagon")


def test_dataset_is_balanced_and_reproducible():
    x1, y1 = make_dataset(n_per_class=10, seed=1)
    x2, y2 = make_dataset(n_per_class=10, seed=1)
    assert x1.shape == (30, 64, 64)
    assert np.bincount(y1).tolist() == [10, 10, 10]
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
