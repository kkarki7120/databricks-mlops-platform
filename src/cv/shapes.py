"""Synthetic image dataset for the computer-vision pipeline.

Generates grayscale images of circles, squares and triangles at random sizes and
positions, with pixel noise. No download needed, so it runs anywhere (including
Databricks Free Edition serverless) and is fully reproducible from a seed.
"""
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle"]


def make_shape_image(shape: str, size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a (size, size) uint8 image containing one shape."""
    if shape not in CLASSES:
        raise ValueError(f"Unknown shape '{shape}', expected one of {CLASSES}")
    rng = rng or np.random.default_rng()

    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    r = int(rng.integers(size // 8, size // 3))
    cx = int(rng.integers(r, size - r))
    cy = int(rng.integers(r, size - r))
    fill = int(rng.integers(150, 256))

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0, 0.08, arr.shape), 0, 1)
    return (arr * 255).astype(np.uint8)


def make_dataset(n_per_class: int = 300, size: int = 64, seed: int = 42):
    """Return (images uint8 [N, size, size], labels int [N])."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label, shape in enumerate(CLASSES):
        for _ in range(n_per_class):
            images.append(make_shape_image(shape, size, rng))
            labels.append(label)
    return np.stack(images), np.array(labels)
