"""Evaluate liver models on CHAOS CT at native resolution.

* ``--model v2``   : the retrained U-Net, on its 5 held-out **test patients** only.
* ``--model original`` : the SegNet from the original notebook (trained on LiTS).
  None of the 20 CHAOS patients was in its training data, so it is evaluated on
  the same 5 test patients (and, for reference, on all 20). Its training data
  mixed slice orientations, so it is scored in all four flip orientations and
  the best one is reported (an optimistic choice in its favour).

Metrics (masks upsampled from 128x128 to 512x512 and compared with the expert
masks): per-patient 3-D Dice and IoU, global Dice/IoU over all pixels, mean
per-slice Dice on slices containing liver, false-positive rate on liver-free
slices, and, for comparison, the notebook's per-pixel "dice" metric.

    python scripts/evaluate_chaos.py --model v2
    python scripts/evaluate_chaos.py --model original
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chaos_data import PATIENTS, full_mask, load  # noqa: E402
from liverlenz.inference import LiverSegmenter  # noqa: E402
from train import split_patients  # noqa: E402

MODELS = {"v2": ROOT / "models" / "liverlenz_unet_v2.onnx",
          "original": ROOT / "models" / "liver_segmentation_model.onnx"}
FLIPS = {"none": lambda a: a, "ud": lambda a: a[..., ::-1, :, :] if a.ndim == 4 else a[..., ::-1, :],
         "lr": lambda a: a[..., :, ::-1, :] if a.ndim == 4 else a[..., :, ::-1],
         "both": lambda a: a[..., ::-1, ::-1, :] if a.ndim == 4 else a[..., ::-1, ::-1]}


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def notebook_dice(y128, prob128, smooth=1.0):
    """Metric the notebook reported for U-Net/SegNet: Dice per pixel (last axis), averaged."""
    return float(np.mean((2 * y128 * prob128 + smooth) / (y128 + prob128 + smooth)))


def score_patient(seg, d, flip="none"):
    f = FLIPS[flip]
    x = np.ascontiguousarray(f(d["x"]))
    prob = np.ascontiguousarray(f(seg.predict_batch(list(x), batch_size=16)))  # back to original orientation
    h, w = d["shape"]
    inter = psum = gsum = 0
    slice_d, fp, empty, nb = [], 0, 0, []
    for i in range(len(prob)):
        pred = cv2.resize(prob[i], (int(w), int(h)), interpolation=cv2.INTER_LINEAR) > 0.5
        gt = full_mask(d, i).astype(bool)
        inter += np.logical_and(pred, gt).sum(); psum += pred.sum(); gsum += gt.sum()
        if gt.any():
            slice_d.append(dice(pred, gt))
        else:
            empty += 1
            fp += int(pred.sum() > 100)
        nb.append(notebook_dice(d["y"][i].astype(np.float32), prob[i]))
    return {"slices": int(len(prob)), "liver_slices": len(slice_d),
            "dice": float(2 * inter / max(psum + gsum, 1)), "iou": float(inter / max(psum + gsum - inter, 1)),
            "_inter": int(inter), "_psum": int(psum), "_gsum": int(gsum),
            "slice_dice": slice_d, "fp": fp, "empty": empty, "nb": nb}


def summarise(rows):
    d = np.array([r["dice"] for r in rows]); j = np.array([r["iou"] for r in rows])
    inter = sum(r["_inter"] for r in rows); ps = sum(r["_psum"] for r in rows); gs = sum(r["_gsum"] for r in rows)
    sd = [v for r in rows for v in r["slice_dice"]]
    return {
        "patients": [r["patient"] for r in rows],
        "dice_per_patient_mean": float(d.mean()), "dice_per_patient_std": float(d.std()),
        "iou_per_patient_mean": float(j.mean()), "iou_per_patient_std": float(j.std()),
        "dice_global": float(2 * inter / max(ps + gs, 1)), "iou_global": float(inter / max(ps + gs - inter, 1)),
        "slice_dice_mean_liver_slices": float(np.mean(sd)) if sd else None,
        "false_positive_rate_liver_free_slices": float(sum(r["fp"] for r in rows) / max(sum(r["empty"] for r in rows), 1)),
        "notebook_style_pixelwise_dice": float(np.mean([v for r in rows for v in r["nb"]])),
        "slices": int(sum(r["slices"] for r in rows)),
        "per_patient": [{k: r[k] for k in ("patient", "slices", "liver_slices", "dice", "iou")} for r in rows],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="v2")
    args = ap.parse_args()
    _, _, test_p = split_patients()
    seg = LiverSegmenter(MODELS[args.model])
    out = {"model": args.model, "dataset": "CHAOS CT (Zenodo 3431873)", "test_patients": test_p, "threshold": 0.5}

    if args.model == "v2":
        rows = []
        for p in test_p:
            r = score_patient(seg, load(p)); r["patient"] = p; rows.append(r)
            print(f"patient {p:2d}: Dice {r['dice']:.3f}  IoU {r['iou']:.3f}", flush=True)
        out["test"] = summarise(rows)
    else:
        by_flip = {}
        for flip in FLIPS:
            rows = []
            for p in PATIENTS:
                r = score_patient(seg, load(p), flip); r["patient"] = p; rows.append(r)
            by_flip[flip] = rows
            print(f"flip {flip:5s}: mean Dice all 20 = {np.mean([r['dice'] for r in rows]):.3f}", flush=True)
        best = max(by_flip, key=lambda k: np.mean([r["dice"] for r in by_flip[k] if r["patient"] in test_p]))
        out["best_orientation"] = best
        out["test"] = summarise([r for r in by_flip[best] if r["patient"] in test_p])
        out["all_20_patients"] = summarise(by_flip[best])
        out["mean_dice_by_orientation_all_20"] = {k: float(np.mean([r["dice"] for r in v])) for k, v in by_flip.items()}

    dest = ROOT / "docs" / "results" / f"chaos_eval_{args.model}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    t = out["test"]
    print(f"\n{args.model} on test patients {test_p}:\n"
          f"  Dice per patient {t['dice_per_patient_mean']:.3f} +/- {t['dice_per_patient_std']:.3f}\n"
          f"  IoU  per patient {t['iou_per_patient_mean']:.3f} +/- {t['iou_per_patient_std']:.3f}\n"
          f"  global Dice {t['dice_global']:.3f}, IoU {t['iou_global']:.3f}\n"
          f"  per-slice Dice (liver slices) {t['slice_dice_mean_liver_slices']}\n"
          f"  FP rate on liver-free slices {t['false_positive_rate_liver_free_slices']:.3f}\n"
          f"  notebook-style per-pixel 'dice' {t['notebook_style_pixelwise_dice']:.3f}\nSaved {dest}")


if __name__ == "__main__":
    main()
