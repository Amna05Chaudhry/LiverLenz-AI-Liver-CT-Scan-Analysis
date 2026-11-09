"""Build the demo slices in examples/ from held-out CHAOS test patients.

Picks one slice per liver level (large / medium / small cross-section), each
from a different test patient, choosing the slice with the *median* v2 Dice in
its band (representative, not cherry-picked). Full-resolution DICOMs are
fetched from the CHAOS ZIP and saved as model-ready 3-channel JPGs. The
original repo sample (a LiTS pelvis slice with no liver) is kept as a
negative example.

    python scripts/make_examples.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chaos_data import URL, load  # noqa: E402
from liverlenz.inference import LiverSegmenter  # noqa: E402
from liverlenz.preprocess import hu_to_rgb  # noqa: E402
from train import split_patients  # noqa: E402

OUT = ROOT / "examples"
BANDS = [("Mid-liver, large section", 0.12, 1.0), ("Liver and kidney level", 0.05, 0.12),
         ("Upper / lower liver", 0.015, 0.05)]


def slice_dice(seg, d):
    prob = seg.predict_batch(list(d["x"]), batch_size=16) > 0.5
    y = d["y"].astype(bool)
    out = []
    for p, g in zip(prob, y):
        s = p.sum() + g.sum()
        out.append(2 * np.logical_and(p, g).sum() / s if s else 1.0)
    return np.array(out), y.mean((1, 2))


def main():
    _, _, test_p = split_patients()
    seg = LiverSegmenter(ROOT / "models" / "liverlenz_unet_v2.onnx")
    cands = []
    for p in test_p:
        d = load(p)
        dices, frac = slice_dice(seg, d)
        cands += [(p, i, float(dices[i]), float(frac[i])) for i in range(len(frac))]

    chosen, used = [], set()
    for title, lo, hi in BANDS:
        pool = sorted([c for c in cands if lo <= c[3] < hi and c[0] not in used], key=lambda c: c[2])
        if pool:
            c = pool[len(pool) // 2]
            used.add(c[0])
            chosen.append((title, *c))

    import pydicom

    from liverlenz.remote_zip import HTTPRangeFile

    z = zipfile.ZipFile(HTTPRangeFile(URL, block=1 << 20))
    names = z.namelist()
    meta = []
    for title, p, i, dsc, frac in chosen:
        dcm = sorted(n for n in names if f"/CT/{p}/DICOM_anon/" in n and n.endswith(".dcm"))
        ds = pydicom.dcmread(io.BytesIO(z.read(dcm[i])))
        hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        name = f"chaos_p{p:02d}_s{i:03d}.jpg"
        Image.fromarray(hu_to_rgb(hu)).save(OUT / name, quality=92)
        meta.append({"file": name, "title": title, "pixel_spacing": round(float(ds.PixelSpacing[0]), 3),
                     "description": f"CHAOS CT patient {p}, slice {i}: held-out test patient. "
                                    f"Dice vs expert mask on this slice: {dsc:.2f}.",
                     "dice": round(dsc, 3), "expect_liver": True})
        print(f"{name}: {title}, liver {100 * frac:.1f}% of slice, Dice {dsc:.3f}")
    meta.append({"file": "lits_v101_s204_pelvis.jpg", "title": "Pelvis (no liver)", "pixel_spacing": None,
                 "description": "The original repo sample: LiTS volume 101, slice 204. A pelvic slice with no liver, "
                                "useful to check for false positives. Shown spine-up, as exported by the LiTS loader.",
                 "dice": None, "expect_liver": False})
    (OUT / "examples.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(meta)} examples to {OUT}")


if __name__ == "__main__":
    main()
