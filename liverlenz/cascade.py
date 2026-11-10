"""LiverLenz v2: two-stage liver -> tumour segmentation (ONNX Runtime, CPU).

Mirrors ``notebooks/train_lits_cascade.py`` exactly:

* intensities: HU clipped to [-100, 250] and scaled to 0..255 (``window_u8``);
* 2.5-D input: channels = slices k-1, k, k+1 (a single 2-D image is repeated 3x, a mode the
  model was trained on);
* stage 1 (liver) on the whole slice resized to 256; stage 2 (tumour) on a square crop around the
  liver taken from the 512 x 512 slice and resized to 256;
* tumour pixels are kept only inside the liver; separate lesions are connected components.

Inputs can be a NIfTI volume, a DICOM file, or an ordinary image (PNG/JPG).
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
HU_MIN, HU_MAX = -100.0, 250.0
S1, S2, BASE = 256, 256, 512
CROP_MARGIN = 16
LIVER_RGB = (20, 184, 166)     # teal
TUMOUR_RGB = (239, 68, 68)     # red
BOX_RGB = (250, 204, 21)       # amber


# ------------------------------------------------------------ input loading
def window_u8(hu: np.ndarray) -> np.ndarray:
    return ((np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)


@dataclass
class Scan:
    """A CT input normalised to (Z, 512, 512) uint8 windowed slices in radiological display."""
    slices: np.ndarray
    spacing_mm: tuple[float, float, float] | None   # (z, y, x) at 512 x 512; None if unknown
    kind: str                                        # "volume" | "dicom" | "image"
    name: str
    original_size: tuple[int, int]

    @property
    def is_volume(self) -> bool:
        return len(self.slices) > 1


def _to_512(img: np.ndarray, interp=cv2.INTER_AREA) -> np.ndarray:
    return img if img.shape == (BASE, BASE) else cv2.resize(img, (BASE, BASE), interpolation=interp)


def load_nifti(data: bytes | str | Path, name: str = "volume.nii") -> Scan:
    import nibabel as nib

    if isinstance(data, (bytes, bytearray)):
        fh = nib.FileHolder(fileobj=io.BytesIO(data))
        if name.endswith(".gz"):
            import gzip
            fh = nib.FileHolder(fileobj=io.BytesIO(gzip.decompress(data)))
        img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    else:
        img = nib.load(str(data))
    img = nib.as_closest_canonical(img)
    a = np.asarray(img.dataobj, dtype=np.float32)
    a = a[::-1, ::-1].transpose(1, 0, 2)                 # RAS -> radiological display
    h, w = a.shape[:2]
    z, sx, sy = (float(v) for v in (img.header.get_zooms()[2], *img.header.get_zooms()[:2]))
    slices = np.stack([_to_512(window_u8(a[..., k])) for k in range(a.shape[2])])
    return Scan(slices, (z, sy * h / BASE, sx * w / BASE), "volume", name, (w, h))


def load_dicom(data: bytes, name: str = "slice.dcm") -> Scan:
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data))
    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    h, w = hu.shape
    ps = [float(v) for v in getattr(ds, "PixelSpacing", [0, 0])]
    spacing = (float(getattr(ds, "SliceThickness", 0) or 0), ps[0] * h / BASE, ps[1] * w / BASE) if ps[0] else None
    return Scan(_to_512(window_u8(hu))[None], spacing, "dicom", name, (w, h))


def load_image(data, name: str = "image.png", pixel_spacing_mm: float | None = None) -> Scan:
    """An ordinary CT picture. Pixel intensities are used as they are (no HU available)."""
    if isinstance(data, (bytes, bytearray)):
        data = Image.open(io.BytesIO(data))
    if isinstance(data, (str, Path)):
        data = Image.open(data)
    if isinstance(data, Image.Image):
        g = np.array(data.convert("L"))
    else:
        g = np.asarray(data)
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) if g.ndim == 3 else g
    h, w = g.shape
    sp = None if not pixel_spacing_mm else (0.0, pixel_spacing_mm * h / BASE, pixel_spacing_mm * w / BASE)
    return Scan(_to_512(g.astype(np.uint8))[None], sp, "image", name, (w, h))


def load_any(data: bytes, name: str, pixel_spacing_mm: float | None = None) -> Scan:
    n = name.lower()
    if n.endswith((".nii", ".nii.gz")):
        return load_nifti(data, n)
    if n.endswith(".dcm") or data[128:132] == b"DICM":
        return load_dicom(data, name)
    return load_image(data, name, pixel_spacing_mm)


# --------------------------------------------------------------- the model
def stack25(slices: np.ndarray, k: int) -> np.ndarray:
    n = len(slices)
    return np.stack([slices[max(k - 1, 0)], slices[k], slices[min(k + 1, n - 1)]], 0)


def liver_box(mask: np.ndarray, margin: int = CROP_MARGIN):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(min(max(max(y1 - y0, x1 - x0) + 2 * margin, 96), BASE))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return (int(np.clip(cy - side // 2, 0, BASE - side)), int(np.clip(cx - side // 2, 0, BASE - side)), side)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        if n <= 1:
            return mask
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        return lab == (1 + int(np.argmax(sizes)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    return lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


@dataclass
class Lesion:
    id: int
    bbox: list[int]                 # [x, y, w, h] in original-image pixels (on the reported slice)
    slice: int
    area_px: int
    area_mm2: float | None
    max_diameter_mm: float | None
    volume_ml: float | None
    slices: list[int]               # first and last slice the lesion appears on
    mean_prob: float
    centroid: list[int]


@dataclass
class CascadeResult:
    liver: np.ndarray               # (Z, 512, 512) bool
    tumour: np.ndarray              # (Z, 512, 512) bool
    tumour_prob: np.ndarray         # (Z, 512, 512) float32
    lesions: list[Lesion] = field(default_factory=list)
    key_slice: int = 0
    lesion_map: np.ndarray | None = None  # (Z, 512, 512) int16: lesion id per voxel (0 = none)


class LiverTumourCascade:
    def __init__(self, models_dir: str | Path | None = None, threshold: float = 0.5):
        self.dir = Path(models_dir or os.getenv("LIVERLENZ_MODELS", MODELS))
        self.threshold = threshold
        self._s1 = self._s2 = None
        meta = self.dir / "model_meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

    @property
    def available(self) -> bool:
        return (self.dir / "liver_stage1.onnx").exists() and (self.dir / "tumour_stage2.onnx").exists()

    def _sessions(self):
        if self._s1 is None:
            import onnxruntime as ort
            opts = ort.SessionOptions(); opts.log_severity_level = 3
            opts.intra_op_num_threads = int(os.getenv("LIVERLENZ_THREADS", "4"))
            self._s1 = ort.InferenceSession(str(self.dir / "liver_stage1.onnx"), opts, providers=["CPUExecutionProvider"])
            self._s2 = ort.InferenceSession(str(self.dir / "tumour_stage2.onnx"), opts, providers=["CPUExecutionProvider"])
        return self._s1, self._s2

    def run(self, scan: Scan, batch: int = 1) -> CascadeResult:
        s1, s2 = self._sessions()
        x = scan.slices
        Z = len(x)
        liver_p = np.zeros((Z, BASE, BASE), np.float32)
        for k in range(Z):  # batch size 1: low, constant memory
            inp = np.stack([cv2.resize(c, (S1, S1), interpolation=cv2.INTER_AREA) for c in stack25(x, k)])[None]
            p = s1.run(None, {"input": inp.astype(np.float32) / 255.0})[0][0, 0]
            liver_p[k] = cv2.resize(p, (BASE, BASE), interpolation=cv2.INTER_LINEAR)
        liver = _largest_component(liver_p > self.threshold)
        tum_p = np.zeros((Z, BASE, BASE), np.float32)
        for k in range(Z):
            box = liver_box(liver[k])
            if box is None:
                continue
            y0, x0, s = box
            crop = stack25(x, k)[:, y0:y0 + s, x0:x0 + s]
            inp = np.stack([cv2.resize(c, (S2, S2), interpolation=cv2.INTER_LINEAR) for c in crop])[None]
            p = s2.run(None, {"input": inp.astype(np.float32) / 255.0})[0][0, 0]
            tum_p[k, y0:y0 + s, x0:x0 + s] = cv2.resize(p, (s, s), interpolation=cv2.INTER_LINEAR)
        tumour = (tum_p > self.threshold) & liver
        res = CascadeResult(liver, tumour, tum_p)
        res.lesions = self._lesions(res, scan)
        if res.lesions:
            res.key_slice = res.lesions[0].slice
        elif liver.any():
            res.key_slice = int(np.argmax(liver.sum((1, 2))))
        return res

    def _lesions(self, res: CascadeResult, scan: Scan, min_px: int = 20) -> list[Lesion]:
        from scipy import ndimage

        sx = scan.original_size[0] / BASE
        sy = scan.original_size[1] / BASE
        sp = scan.spacing_mm
        lab, n = ndimage.label(res.tumour)
        out, raw = [], {}
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = lab[sl] == i
            if comp.sum() < min_px:
                continue
            areas = comp.sum((1, 2))
            kz = int(np.argmax(areas))
            m = comp[kz]
            ys, xs = np.nonzero(m)
            y0, x0 = ys.min() + sl[1].start, xs.min() + sl[2].start
            h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            zabs = kz + sl[0].start
            area_mm2 = diam = vol = None
            if sp and sp[1] > 0:
                area_mm2 = float(m.sum() * sp[1] * sp[2])
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32) * np.array([sp[2], sp[1]], np.float32)
                if len(pts) > 1:
                    hull = cv2.convexHull(pts).reshape(-1, 2)
                    d = np.sqrt(((hull[:, None] - hull[None]) ** 2).sum(-1))
                    diam = float(d.max())
                if scan.is_volume and sp[0] > 0:
                    vol = float(comp.sum() * sp[0] * sp[1] * sp[2] / 1000.0)
            raw[len(out)] = i
            out.append(Lesion(
                id=0, bbox=[int(x0 * sx), int(y0 * sy), max(1, int(w * sx)), max(1, int(h * sy))], slice=int(zabs),
                area_px=int(round(m.sum() * sx * sy)), area_mm2=None if area_mm2 is None else round(area_mm2, 1),
                max_diameter_mm=None if diam is None else round(diam, 1), volume_ml=None if vol is None else round(vol, 2),
                slices=[int(sl[0].start), int(sl[0].stop - 1)], mean_prob=round(float(res.tumour_prob[sl][comp].mean()), 3),
                centroid=[int((xs.mean() + sl[2].start) * sx), int((ys.mean() + sl[1].start) * sy)]))
        order = sorted(range(len(out)), key=lambda q: -(out[q].volume_ml or out[q].area_px))
        res.lesion_map = np.zeros(lab.shape, np.int16)
        lut = np.zeros(n + 1, np.int16)
        for j, q in enumerate(order, 1):
            out[q].id = j
            lut[raw[q]] = j
        res.lesion_map = lut[lab]
        return [out[q] for q in order]


# ----------------------------------------------------------- summary/render
def slice_stats(scan: Scan, res: CascadeResult, k: int) -> dict:
    liver_px, tum_px = int(res.liver[k].sum()), int(res.tumour[k].sum())
    sp = scan.spacing_mm
    px_mm2 = sp[1] * sp[2] if sp and sp[1] > 0 else None
    lesions_here = [asdict(l) for l in res.lesions if l.slices[0] <= k <= l.slices[1]]
    return {
        "slice": k, "n_slices": len(scan.slices),
        "liver_detected": liver_px > 0, "liver_area_pct": round(100 * liver_px / BASE ** 2, 2),
        "liver_area_cm2": round(liver_px * px_mm2 / 100, 1) if px_mm2 else None,
        "tumour_detected": tum_px > 0, "tumour_area_px": int(tum_px * (scan.original_size[0] / BASE) * (scan.original_size[1] / BASE)),
        "tumour_area_cm2": round(tum_px * px_mm2 / 100, 2) if px_mm2 else None,
        "tumour_burden_pct": round(100 * tum_px / liver_px, 1) if liver_px else 0.0,
        "lesions_on_slice": lesions_here,
    }


def scan_summary(scan: Scan, res: CascadeResult) -> dict:
    sp = scan.spacing_mm
    vox_ml = sp[0] * sp[1] * sp[2] / 1000 if scan.is_volume and sp and sp[0] > 0 else None
    return {
        "kind": scan.kind, "name": scan.name, "n_slices": len(scan.slices), "original_size": list(scan.original_size),
        "spacing_mm": None if not sp else [round(v, 3) for v in sp],
        "liver_volume_ml": round(float(res.liver.sum() * vox_ml), 1) if vox_ml else None,
        "tumour_volume_ml": round(float(res.tumour.sum() * vox_ml), 2) if vox_ml else None,
        "tumour_burden_pct": round(100 * res.tumour.sum() / res.liver.sum(), 2) if res.liver.any() else 0.0,
        "n_lesions": len(res.lesions), "lesions": [asdict(l) for l in res.lesions], "key_slice": res.key_slice,
        "liver_detected": bool(res.liver.any()), "tumour_detected": bool(res.tumour.any()),
    }


def _to_original(img: np.ndarray, scan: Scan) -> np.ndarray:
    w, h = scan.original_size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) if (w, h) != (BASE, BASE) else img


HEAT_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0, ::-1]  # RGB


def _contours(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts


def render(scan: Scan, res: CascadeResult, k: int) -> dict[str, np.ndarray]:
    """Images for slice k at the original size.

    * ``original``: the windowed slice;
    * ``overlay``: liver (teal tint + outline) and tumour (red-orange fill + yellow outline), baked in;
    * ``boxes``: lesion bounding boxes numbered like the lesion table;
    * ``layer_liver`` / ``layer_tumour`` / ``layer_heat``: transparent RGBA layers for the
      interactive viewer (toggles and opacity are applied in the browser).
    """
    g = scan.slices[k]
    base = np.stack([g] * 3, -1)
    lv, tm = res.liver[k], res.tumour[k]
    t = max(1, round(BASE / 256))

    ov = base.astype(np.float32)
    ov[lv] = 0.72 * ov[lv] + 0.28 * np.array(LIVER_RGB)
    ov[tm] = 0.25 * ov[tm] + 0.75 * np.array(TUMOUR_RGB)
    ov = ov.astype(np.uint8)
    cv2.drawContours(ov, _contours(lv), -1, LIVER_RGB, t)
    cv2.drawContours(ov, _contours(tm), -1, BOX_RGB, t)

    boxes = base.copy()
    cv2.drawContours(boxes, _contours(lv), -1, LIVER_RGB, 1)
    lm = res.lesion_map[k] if res.lesion_map is not None else tm.astype(np.int16)
    for lid in [int(v) for v in np.unique(lm) if v]:
        ys, xs = np.nonzero(lm == lid)
        x0, y0, x1, y1 = xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 3
        cv2.rectangle(boxes, (x0, y0), (x1, y1), BOX_RGB, 2)
        label = str(lid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y0 - 4 if y0 - th - 6 > 0 else y1 + th + 4
        cv2.rectangle(boxes, (x0, ty - th - 3), (x0 + tw + 5, ty + 2), BOX_RGB, -1)
        cv2.putText(boxes, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    def rgba(mask, rgb, alpha, outline=None, outline_rgb=None):
        a = np.zeros((BASE, BASE, 4), np.uint8)
        a[mask] = (*rgb, alpha)
        if outline:
            cv2.drawContours(a, _contours(mask), -1, (*(outline_rgb or rgb), 255), outline)
        return a

    layer_liver = rgba(lv, LIVER_RGB, 90, t)
    layer_tumour = rgba(tm, TUMOUR_RGB, 215, t + 1, BOX_RGB)
    heat = np.zeros((BASE, BASE, 4), np.uint8)
    if res.tumour_prob is not None:
        p = np.asarray(res.tumour_prob[k], np.float32)
        show = p > 0.1
        idx = np.clip(p * 255, 0, 255).astype(np.uint8)
        heat[..., :3] = HEAT_LUT[idx]
        heat[..., 3] = np.where(show, np.clip(60 + 195 * p, 0, 255), 0).astype(np.uint8)
    out = {"original": base, "overlay": ov, "boxes": boxes,
           "layer_liver": layer_liver, "layer_tumour": layer_tumour, "layer_heat": heat}
    return {k2: _to_original(v, scan) for k2, v in out.items()}


def to_png_base64(rgb: np.ndarray) -> str:
    import base64
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
