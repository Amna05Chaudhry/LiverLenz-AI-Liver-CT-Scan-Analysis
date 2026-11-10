"""Build demo inputs in examples/ from held-out test patients (MSD Task03 = LiTS, CC BY-SA 4.0).

For each case downloaded by ``scripts/fetch_msd_examples.py`` this writes:

* ``lits_<id>.nii.gz``: a crop of up to 40 slices around the largest tumour (int16, original affine),
  small enough to ship with the repo and to run quickly on CPU;
* ``lits_<id>_thumb.png``: thumbnail of the key slice (liver window);
* for the first case also ``lits_<id>_slice.png``: one plain greyscale PNG slice, to demo image upload.

All cases are from the notebook's held-out test split (seed 42), so the demo never shows training data.

    python scripts/fetch_msd_examples.py 0 33 37
    python scripts/make_examples.py 0 33 37
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "data" / "msd", ROOT / "examples"
TEST = {0, 2, 3, 4, 20, 24, 25, 33, 37, 51, 56, 68, 70, 80, 90, 91, 92, 101, 106, 111}
HALF = 20


def display(a):  # RAS -> radiological display
    return a[::-1, ::-1].T


def window(hu, lo=-100, hi=250):
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def main(ids):
    OUT.mkdir(exist_ok=True)
    meta = [m for m in (json.loads((OUT / "examples.json").read_text()) if (OUT / "examples.json").exists() else [])
            if not m["file"].startswith("lits_")]
    for n, cid in enumerate(ids):
        assert cid in TEST, f"case {cid} is not in the held-out test split"
        img = nib.as_closest_canonical(nib.load(SRC / f"image_liver_{cid}.nii.gz"))
        lab = nib.as_closest_canonical(nib.load(SRC / f"label_liver_{cid}.nii.gz"))
        seg = np.asarray(lab.dataobj).astype(np.uint8)
        tum = (seg == 2).sum((0, 1))
        k = int(np.argmax(tum)) if tum.any() else int(np.argmax((seg > 0).sum((0, 1))))
        z0, z1 = max(0, k - HALF), min(seg.shape[2], k + HALF)
        crop = img.slicer[:, :, z0:z1]
        data = np.asarray(crop.dataobj).astype(np.int16)
        out = nib.Nifti1Image(data, crop.affine)
        out.header.set_zooms(crop.header.get_zooms()[:3])
        name = f"lits_{cid}.nii.gz"
        nib.save(out, OUT / name)
        key = display(np.asarray(img.dataobj[:, :, k], dtype=np.float32))
        Image.fromarray(window(key)).resize((256, 256)).save(OUT / f"lits_{cid}_thumb.png")
        n_les = int((seg[..., z0:z1] == 2).any())
        sp = [round(float(v), 3) for v in crop.header.get_zooms()[:3]]
        meta.append({"file": name, "thumb": f"lits_{cid}_thumb.png", "title": f"LiTS case {cid} (volume)",
                     "description": f"Held-out test patient {cid}: {z1 - z0} axial slices around the largest tumour, "
                                    f"{sp[0]} x {sp[1]} mm pixels, {sp[2]} mm slices. From MSD Task03 (LiTS), CC BY-SA 4.0.",
                     "pixel_spacing": None, "expect_liver": True, "expect_tumour": bool(tum.any())})
        if n == 0:
            png = f"lits_{cid}_slice.png"
            Image.fromarray(window(key)).save(OUT / png)
            meta.append({"file": png, "thumb": png, "title": f"LiTS case {cid} (single PNG)",
                         "description": "The same patient's key slice as a plain greyscale PNG, to show image upload.",
                         "pixel_spacing": sp[0], "expect_liver": True, "expect_tumour": bool(tum.any())})
        print(f"{name}: slices {z0}-{z1 - 1} of {seg.shape[2]}, key slice {k}, tumour px on key {int(tum[k])}, "
              f"{(OUT / name).stat().st_size / 1e6:.1f} MB")
    (OUT / "examples.json").write_text(json.dumps(meta, indent=2))
    print(f"examples.json: {[m['file'] for m in meta]}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0])
