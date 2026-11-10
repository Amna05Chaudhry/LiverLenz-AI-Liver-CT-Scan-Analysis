"""Held-out test evaluation of the exported ONNX cascade (CPU, batch size 1).

Same protocol as section 6 of notebooks/train_lits_cascade.ipynb:
* the 20 test patients of the seed-42 patient-wise split (never used for training or model selection);
* each volume cropped to the liver's slice range +-12 slices (as the notebook caches it);
* liver and tumour Dice / IoU per volume and pooled over all test voxels;
* lesions = 3-D connected components of the expert tumour mask; detected if the prediction overlaps it;
  recall overall and for lesions >= 10 mm equivalent diameter; precision over predicted components.

Test scans are read from the Medical Segmentation Decathlon copy of LiTS (liver_N == LiTS volume-N):

    python scripts/fetch_msd_examples.py 0 2 3 4 20 24 25 33 37 51 56 68 70 80 90 91 92 101 106 111
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liverlenz.cascade import BASE, LiverTumourCascade, Scan, window_u8  # noqa: E402

TEST = [0, 2, 3, 4, 20, 24, 25, 33, 37, 51, 56, 68, 70, 80, 90, 91, 92, 101, 106, 111]
Z_MARGIN = 12
SRC = ROOT / "data" / "msd"


def display(a):
    return np.ascontiguousarray(a[::-1, ::-1].transpose(1, 0, 2))


def load_case(cid: int):
    img = nib.as_closest_canonical(nib.load(SRC / f"image_liver_{cid}.nii.gz"))
    lab = nib.as_closest_canonical(nib.load(SRC / f"label_liver_{cid}.nii.gz"))
    seg = display(np.asarray(lab.dataobj).astype(np.uint8))
    has = np.where(seg.reshape(-1, seg.shape[2]).max(0) > 0)[0]
    z0, z1 = max(0, has[0] - Z_MARGIN), min(seg.shape[2], has[-1] + Z_MARGIN + 1)
    vol = display(np.asarray(img.slicer[:, :, z0:z1].dataobj, dtype=np.float32))
    seg = seg[:, :, z0:z1]
    h, w = vol.shape[:2]
    sl = np.stack([cv2.resize(window_u8(vol[..., k]), (BASE, BASE), interpolation=cv2.INTER_AREA) if (h, w) != (BASE, BASE)
                   else window_u8(vol[..., k]) for k in range(vol.shape[2])])
    gt = np.stack([cv2.resize(seg[..., k], (BASE, BASE), interpolation=cv2.INTER_NEAREST) if (h, w) != (BASE, BASE)
                   else seg[..., k] for k in range(seg.shape[2])])
    zooms = img.header.get_zooms()
    sp = (float(zooms[2]), float(zooms[1]) * h / BASE, float(zooms[0]) * w / BASE)
    del vol
    return Scan(sl, sp, "volume", f"liver_{cid}", (w, h)), gt


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    model = LiverTumourCascade()
    rows, pooled = [], dict(li=0, lp=0, lg=0, ti=0, tp=0, tg=0)
    les = dict(gt=0, det=0, gt10=0, det10=0, pred=0, pred_tp=0)
    t_all = time.time()
    for cid in TEST:
        if not ((SRC / f"image_liver_{cid}.nii.gz").exists() and (SRC / f"label_liver_{cid}.nii.gz").exists()):
            print(f"case {cid}: missing, skipped")
            continue
        t0 = time.time()
        scan, gt_lab = load_case(cid)
        res = model.run(scan)
        liver, tumour = res.liver, res.tumour
        gl, gtm = gt_lab > 0, gt_lab == 2
        r = {"volume": cid, "slices": int(len(gt_lab)), "liver_dice": dice(liver, gl), "liver_iou": iou(liver, gl),
             "has_tumour": bool(gtm.any())}
        if gtm.any():
            r["tumour_dice"], r["tumour_iou"] = dice(tumour, gtm), iou(tumour, gtm)
        pooled["li"] += int((liver & gl).sum()); pooled["lp"] += int(liver.sum()); pooled["lg"] += int(gl.sum())
        pooled["ti"] += int((tumour & gtm).sum()); pooled["tp"] += int(tumour.sum()); pooled["tg"] += int(gtm.sum())
        vox_mm3 = float(np.prod(scan.spacing_mm))
        glab, gn = ndimage.label(gtm)
        for i, sl in enumerate(ndimage.find_objects(glab), start=1):
            comp = glab[sl] == i
            d_eq = (6 * comp.sum() * vox_mm3 / math.pi) ** (1 / 3)
            hit = bool((comp & tumour[sl]).any())
            les["gt"] += 1; les["det"] += hit
            if d_eq >= 10:
                les["gt10"] += 1; les["det10"] += hit
        plab, pn = ndimage.label(tumour)
        for i, sl in enumerate(ndimage.find_objects(plab), start=1):
            les["pred"] += 1; les["pred_tp"] += bool(((plab[sl] == i) & gtm[sl]).any())
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"case {cid:3d}: {r['slices']:3d} slices  liver Dice {r['liver_dice']:.3f}"
              + (f"  tumour Dice {r['tumour_dice']:.3f}" if "tumour_dice" in r else "  (no tumour)") + f"  ({r['seconds']}s)", flush=True)
        del scan, res, gt_lab, glab, plab

    def ms(key):
        v = np.array([x[key] for x in rows if key in x])
        return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)), "n": int(len(v))}

    m = {
        "protocol": "LiTS test split (seed 42, 20 patients), liver range +-12 slices, 512x512, threshold 0.5, ONNX on CPU",
        "test_volumes": [x["volume"] for x in rows],
        "liver_dice_per_volume": ms("liver_dice"), "liver_iou_per_volume": ms("liver_iou"),
        "liver_dice_global": 2 * pooled["li"] / max(pooled["lp"] + pooled["lg"], 1),
        "liver_iou_global": pooled["li"] / max(pooled["lp"] + pooled["lg"] - pooled["li"], 1),
        "tumour_dice_per_volume": ms("tumour_dice"), "tumour_iou_per_volume": ms("tumour_iou"),
        "tumour_dice_global": 2 * pooled["ti"] / max(pooled["tp"] + pooled["tg"], 1),
        "tumour_iou_global": pooled["ti"] / max(pooled["tp"] + pooled["tg"] - pooled["ti"], 1),
        "lesion_recall": les["det"] / max(les["gt"], 1), "lesion_recall_ge10mm": les["det10"] / max(les["gt10"], 1),
        "lesion_precision": les["pred_tp"] / max(les["pred"], 1), "lesion_counts": les,
        "total_seconds": round(time.time() - t_all), "per_volume": rows,
    }
    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v for k, v in m.items() if k != "per_volume"}, indent=1))


if __name__ == "__main__":
    main()
