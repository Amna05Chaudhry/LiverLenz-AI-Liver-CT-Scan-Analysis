"""Liver segmentation inference with the trained Keras SegNet.

The model takes a 128x128 RGB slice (see ``liverlenz.preprocess``) and outputs a
per-pixel liver probability. This module wraps loading, prediction, mask
statistics and the visualisations shown in the web app.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "liver_segmentation_model.onnx"
INPUT_SIZE = (128, 128)
ACCENT_BGR = (118, 118, 15)      # teal #0F766E in BGR
OVERLAY_RGB = (20, 184, 166)     # teal-500 for the mask overlay
BOX_RGB = (250, 204, 21)         # amber box for contrast


@dataclass
class ScanStats:
    image_width: int
    image_height: int
    liver_detected: bool
    liver_area_px: int
    liver_area_pct: float
    bbox: list[int] | None             # [x, y, w, h] in original image pixels
    centroid: list[int] | None         # [x, y]
    region: str
    components: int
    mean_confidence: float | None      # mean liver probability inside the mask
    pixel_spacing_mm: float | None = None
    liver_area_mm2: float | None = None
    bbox_mm: list[float] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentationResult:
    image: np.ndarray        # original RGB uint8
    probability: np.ndarray  # (H, W) float32 in [0, 1], original resolution
    mask: np.ndarray         # (H, W) uint8 {0, 1}
    stats: ScanStats


def dice_coefficient(y_true, y_pred):  # needed only to deserialize the .h5
    import tensorflow as tf

    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def to_rgb_array(image) -> np.ndarray:
    """Accept a path, bytes, PIL image or array and return an RGB uint8 array."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.clip(0, 255).astype(np.uint8)
    return arr


def describe_region(cx: float, cy: float, w: int, h: int) -> str:
    horiz = ["left", "central", "right"][min(int(3 * cx / w), 2)]
    vert = ["upper", "middle", "lower"][min(int(3 * cy / h), 2)]
    return f"{vert} {horiz} part of the image"


def compute_stats(mask: np.ndarray, prob: np.ndarray, pixel_spacing_mm: float | None = None) -> ScanStats:
    h, w = mask.shape
    area = int(mask.sum())
    notes: list[str] = []
    if area == 0:
        return ScanStats(w, h, False, 0, 0.0, None, None, "none", 0, None,
                         pixel_spacing_mm, None, None,
                         ["No liver tissue was segmented on this slice."])

    n_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = n_labels - 1
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    cx, cy = float(xs.mean()), float(ys.mean())
    pct = 100.0 * area / (h * w)
    conf = float(prob[mask.astype(bool)].mean())

    if components > 1:
        notes.append(f"Mask has {components} separate regions; the largest covers "
                     f"{100 * cc_stats[1:, cv2.CC_STAT_AREA].max() / area:.0f}% of the segmented area.")
    if pct < 1.0:
        notes.append("Very small liver area: the slice may be near the liver dome or the inferior tip.")
    if cx > 0.6 * w:
        notes.append("Mask is mostly on the image right. In standard radiological display the liver "
                     "appears on the image left, so check orientation or possible false positives.")

    area_mm2 = bbox_mm = None
    if pixel_spacing_mm:
        area_mm2 = round(area * pixel_spacing_mm ** 2, 1)
        bbox_mm = [round(bbox[2] * pixel_spacing_mm, 1), round(bbox[3] * pixel_spacing_mm, 1)]

    return ScanStats(w, h, True, area, round(pct, 2), bbox, [int(cx), int(cy)],
                     describe_region(cx, cy, w, h), components, round(conf, 3),
                     pixel_spacing_mm, area_mm2, bbox_mm, notes)


def _prep(rgb: np.ndarray) -> np.ndarray:
    # Same as training: PIL resize (bicubic) to 128x128, scale to [0, 1]
    return np.array(Image.fromarray(rgb).resize(INPUT_SIZE), dtype=np.float32) / 255.0


class LiverSegmenter:
    """Loads the model once and segments RGB CT slices.

    Uses the ONNX export with onnxruntime (small, no TensorFlow needed). A Keras
    ``.h5`` path also works if TensorFlow/Keras is installed.
    """

    def __init__(self, model_path: str | Path | None = None, threshold: float = 0.5):
        self.model_path = Path(model_path or os.getenv("LIVERLENZ_MODEL", DEFAULT_MODEL_PATH))
        self.threshold = threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model weights not found at {self.model_path}")
            if self.model_path.suffix == ".onnx":
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                self._model = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
            else:
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                import keras

                self._model = keras.models.load_model(
                    self.model_path, compile=False, custom_objects={"dice_coefficient": dice_coefficient})
        return self._model

    def _run(self, x: np.ndarray) -> np.ndarray:
        m = self.model
        if self.model_path.suffix == ".onnx":
            return m.run(None, {m.get_inputs()[0].name: x})[0][..., 0]
        return np.asarray(m(x, training=False))[..., 0]

    def predict_probability(self, rgb: np.ndarray) -> np.ndarray:
        """Return the 128x128 liver probability map for an RGB uint8 image."""
        return self._run(_prep(rgb)[None])[0]

    def predict_batch(self, rgbs: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        x = np.stack([_prep(r) for r in rgbs])
        return np.concatenate([self._run(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])

    def segment(self, image, pixel_spacing_mm: float | None = None) -> SegmentationResult:
        rgb = to_rgb_array(image)
        h, w = rgb.shape[:2]
        prob_small = self.predict_probability(rgb)
        prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob > self.threshold).astype(np.uint8)
        return SegmentationResult(rgb, prob, mask, compute_stats(mask, prob, pixel_spacing_mm))


# ---------------------------------------------------------------- rendering
def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(OVERLAY_RGB, dtype=np.float32)
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(1, round(min(rgb.shape[:2]) / 256))
    cv2.drawContours(out, contours, -1, OVERLAY_RGB, thickness)
    return out


def draw_bbox(rgb: np.ndarray, stats: ScanStats) -> np.ndarray:
    out = rgb.copy()
    if stats.bbox:
        x, y, w, h = stats.bbox
        t = max(2, round(min(rgb.shape[:2]) / 170))
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), BOX_RGB, t)
        cx, cy = stats.centroid
        cv2.drawMarker(out, (cx, cy), BOX_RGB, cv2.MARKER_CROSS, 6 * t, t)
    return out


def mask_image(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)


def to_png_base64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
