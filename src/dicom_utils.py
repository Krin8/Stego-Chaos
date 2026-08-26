"""Shared DICOM reading / intensity-normalization helpers.

These primitives are used by the dataset classes, the preprocessing
pipeline and the demo scripts, which all need the same
read → Hounsfield-rescale → window → 8-bit conversion chain.
"""

import numpy as np
from PIL import Image, ImageOps

try:
    import pydicom
except ImportError:  # pragma: no cover - convenience for Colab-style envs
    import os
    os.system("pip install pydicom -q")
    import pydicom

# Abdominal soft-tissue windows, in Hounsfield units
ABDOMEN_WINDOW = (40.0, 400.0)  # (level, width)
SOFT_TISSUE_BOUNDS = (-150.0, 250.0)


def read_dicom(path, stop_before_pixels=False):
    return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)


def is_ct(dcm):
    return getattr(dcm, "Modality", None) == "CT"


def to_hounsfield(dcm):
    """Pixel array rescaled to Hounsfield units when the metadata allows it."""
    arr = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dcm, "RescaleIntercept", 0) or 0)
    return arr * slope + intercept


def read_hounsfield(path):
    return to_hounsfield(read_dicom(path))


def window_bounds(level, width):
    return level - width / 2, level + width / 2


def clip_normalize(arr, lower, upper):
    """Clip to [lower, upper] and rescale to [0, 1]."""
    arr = np.clip(arr, lower, upper)
    if upper > lower:
        return (arr - lower) / (upper - lower)
    return np.zeros_like(arr)


def percentile_normalize(arr, low=1, high=99):
    lower, upper = np.percentile(arr, (low, high))
    return clip_normalize(arr, lower, upper)


def to_uint8(arr):
    """Convert a [0, 1] float array to 8-bit."""
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def apply_clahe(arr_uint8, clip_limit=2.0, tile_grid_size=(8, 8)):
    import cv2

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(arr_uint8)


def to_rgb_pil(arr_uint8, equalize=False):
    img = Image.fromarray(arr_uint8, mode="L")
    if equalize:
        img = ImageOps.equalize(img)
    return img.convert("RGB")


def windowed_rgb_pil(dcm, level_width=ABDOMEN_WINDOW, equalize=False):
    """CT slice → windowed 8-bit RGB image."""
    lower, upper = window_bounds(*level_width)
    return to_rgb_pil(to_uint8(clip_normalize(to_hounsfield(dcm), lower, upper)), equalize)


def percentile_rgb_pil(dcm, low=1, high=99, equalize=False, rescale=False):
    """MR slice → percentile-normalized 8-bit RGB image."""
    arr = to_hounsfield(dcm) if rescale else dcm.pixel_array.astype(np.float32)
    return to_rgb_pil(to_uint8(percentile_normalize(arr, low, high)), equalize)


def min_max_rgb_pil(dcm):
    """Raw slice → min/max normalized 8-bit RGB image."""
    arr = dcm.pixel_array.astype(np.float32)
    return to_rgb_pil(to_uint8(clip_normalize(arr, arr.min(), arr.max())))
