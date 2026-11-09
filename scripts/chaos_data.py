"""Download and cache the CHAOS CT liver data (Kavur et al., 2021).

CHAOS "Train_Sets" on Zenodo (record 3431873, CC BY-SA 4.0) contains 20
contrast-enhanced abdominal CT studies with expert liver masks. Only the CT
files are fetched from the 890 MB ZIP, using HTTP range requests, and stored
per patient in ``data/chaos128/pXX.npz``:

* ``x``      (N, 128, 128, 3) uint8: model input (``hu_to_rgb`` then resized)
* ``y``      (N, 128, 128)    uint8: liver mask at 128 x 128
* ``y_full`` (N, 512*512/8)   uint8: full-resolution mask, bit-packed
* ``spacing``: in-plane pixel spacing (mm)

Slices keep the standard DICOM display orientation (patient's right on the
image left, spine at the bottom).

    python scripts/chaos_data.py            # all 20 patients (~790 MB transferred)
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liverlenz.preprocess import hu_to_rgb  # noqa: E402

URL = "https://zenodo.org/api/records/3431873/files/CHAOS_Train_Sets.zip/content"
CACHE = ROOT / "data" / "chaos128"
PATIENTS = [1, 2, 5, 6, 8, 10, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
SIZE = 128


def mask_to_128(m: np.ndarray) -> np.ndarray:
    import cv2

    return (cv2.resize(m.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)


def fetch(patients=PATIENTS) -> None:
    import pydicom

    from liverlenz.remote_zip import HTTPRangeFile

    todo = [p for p in patients if not (CACHE / f"p{p:02d}.npz").exists()]
    if not todo:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    print("Opening the CHAOS archive on Zenodo (HTTP range requests)...", flush=True)
    z = zipfile.ZipFile(HTTPRangeFile(URL, block=1 << 20))
    names = z.namelist()
    for p in todo:
        dcm = sorted(n for n in names if f"/CT/{p}/DICOM_anon/" in n and n.endswith(".dcm"))
        gts = sorted(n for n in names if f"/CT/{p}/Ground/" in n and n.endswith(".png"))
        assert dcm and len(dcm) == len(gts), (p, len(dcm), len(gts))
        n = len(dcm)
        x = np.zeros((n, SIZE, SIZE, 3), np.uint8)
        y = np.zeros((n, SIZE, SIZE), np.uint8)
        y_full = None
        spacing = None
        for i in range(n):
            ds = pydicom.dcmread(io.BytesIO(z.read(dcm[i])))
            hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            g = (np.array(Image.open(io.BytesIO(z.read(gts[i])))) > 0).astype(np.uint8)
            if y_full is None:
                y_full = np.zeros((n, g.size // 8), np.uint8)
                shape = g.shape
            x[i] = np.array(Image.fromarray(hu_to_rgb(hu)).resize((SIZE, SIZE)))
            y[i] = mask_to_128(g)
            y_full[i] = np.packbits(g.reshape(-1))
            spacing = float(ds.PixelSpacing[0])
            print(f"\r  patient {p:2d}: {i + 1}/{n}", end="", flush=True)
        np.savez_compressed(CACHE / f"p{p:02d}.npz", x=x, y=y, y_full=y_full, shape=np.array(shape), spacing=spacing)
        print(flush=True)


def load(p: int) -> dict:
    d = np.load(CACHE / f"p{p:02d}.npz")
    return {k: d[k] for k in d.files}


def full_mask(d: dict, i: int) -> np.ndarray:
    h, w = d["shape"]
    return np.unpackbits(d["y_full"][i])[: h * w].reshape(h, w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, nargs="*", default=PATIENTS)
    fetch(ap.parse_args().patients)
