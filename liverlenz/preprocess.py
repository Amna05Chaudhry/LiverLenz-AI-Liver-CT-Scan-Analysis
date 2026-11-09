"""CT preprocessing that reproduces the training pipeline of the notebook.

The model was trained on 3-channel JPEGs produced by fastai's ``save_jpg`` with
``wins=[liver, custom]`` plus a histogram-equalised channel:

* R: liver window      (width 150, level 30)
* G: custom window     (width 200, level 60)
* B: ``hist_scaled``   (per-slice frequency-histogram scaling)

This module is a NumPy port of that code so the app and scripts do not need
fastai / PyTorch. Raw Hounsfield-unit slices (from NIfTI or DICOM) go in, an
RGB ``uint8`` image identical in spirit to the training JPEGs comes out.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

LIVER_WINDOW = (150, 30)   # (width, level) in HU
CUSTOM_WINDOW = (200, 60)
MODEL_WINDOWS = (LIVER_WINDOW, CUSTOM_WINDOW)


def windowed(hu: np.ndarray, width: float, level: float) -> np.ndarray:
    """Clamp HU values to a window and scale to [0, 1] (fastai ``windowed``)."""
    lo = level - width // 2
    hi = level + width // 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)


def freqhist_bins(x: np.ndarray, n_bins: int = 100) -> np.ndarray:
    """Split the value range so each bin holds roughly the same number of pixels."""
    flat = np.sort(x.reshape(-1))
    t = np.concatenate([[0.001], np.arange(n_bins) / n_bins + 1 / (2 * n_bins), [0.999]])
    idx = (len(flat) * t).astype(np.int64)
    return np.unique(flat[idx])


def hist_scaled(x: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1] with ``freqhist_bins`` (fastai ``hist_scaled``)."""
    brks = freqhist_bins(x)
    if len(brks) < 2:  # constant image
        return np.zeros_like(x, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, len(brks))
    return np.clip(np.interp(x, brks, ys), 0.0, 1.0).astype(np.float32)


def hu_to_rgb(hu: np.ndarray) -> np.ndarray:
    """Convert a 2-D HU slice to the 3-channel ``uint8`` image the model expects."""
    hu = np.asarray(hu, dtype=np.float32)
    channels = [windowed(hu, *w) for w in MODEL_WINDOWS] + [hist_scaled(hu)]
    rgb = np.stack(channels, axis=-1)
    return (rgb * 255).astype(np.uint8)


def jpeg_roundtrip(rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Encode/decode as JPEG like the training data (quality 90)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume as (H, W, slices), rotated like the notebook (np.rot90)."""
    import nibabel as nib  # optional dependency, only needed for volumes

    data = nib.load(str(path)).get_fdata()
    return np.rot90(np.asarray(data))


def save_slice_jpg(hu: np.ndarray, path: str | Path, quality: int = 90) -> Path:
    """Save one HU slice as a model-ready JPEG."""
    path = Path(path).with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hu_to_rgb(hu)).save(path, quality=quality)
    return path
