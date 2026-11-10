"""Choose tumour post-processing on the 10 VALIDATION patients only (never the test set).

Grid: tumour probability threshold x minimum lesion size (equivalent diameter, mm) x flip test-time
augmentation. Objective: mean per-patient tumour Dice; lesion precision/recall and pooled Dice are
reported alongside. The chosen setting is written to models/model_meta.json ("postprocessing"), which
the app and scripts/evaluate.py then use.

    python scripts/fetch_msd_examples.py 13 18 59 65 72 100 104 105 108 125
    python scripts/tune_postprocessing.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_case  # noqa: E402
from liverlenz.cascade import LiverTumourCascade, postprocess_tumour  # noqa: E402

VAL = [13, 18, 59, 65, 72, 100, 104, 105, 108, 125]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_MM = [0, 3, 5, 7, 10]


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def main():
    stats = {}  # (tta, thr, mm) -> accumulators
    for cid in VAL:
        scan, gt = load_case(cid)
        gtm = gt == 2
        glab, gn = ndimage.label(gtm)
        for tta in (False, True):
            res = LiverTumourCascade(tta=tta, tumour_threshold=0.5, min_lesion_mm=0).run(scan)
            for thr, mm in itertools.product(THRESHOLDS, MIN_MM):
                t = postprocess_tumour(res.tumour_prob, res.liver, scan.spacing_mm, thr, mm)
                a = stats.setdefault((tta, thr, mm), dict(d=[], i=0, s=0, gt=0, det=0, pred=0, tp=0))
                if gtm.any():
                    a["d"].append(dice(t, gtm))
                a["i"] += int((t & gtm).sum()); a["s"] += int(t.sum() + gtm.sum())
                a["gt"] += gn
                a["det"] += sum(bool((glab[sl] == j)[t[sl]].any()) for j, sl in enumerate(ndimage.find_objects(glab), 1))
                plab, pn = ndimage.label(t)
                a["pred"] += pn
                a["tp"] += sum(bool(gtm[sl][plab[sl] == j].any()) for j, sl in enumerate(ndimage.find_objects(plab), 1))
            del res
        print(f"val case {cid} done", flush=True)

    rows = []
    for (tta, thr, mm), a in stats.items():
        rows.append({"tta": tta, "threshold": thr, "min_lesion_mm": mm, "dice_per_patient": float(np.mean(a["d"])),
                     "dice_pooled": 2 * a["i"] / max(a["s"], 1), "lesion_recall": a["det"] / max(a["gt"], 1),
                     "lesion_precision": a["tp"] / max(a["pred"], 1)})
    rows.sort(key=lambda r: (-r["dice_per_patient"], -r["lesion_precision"]))
    best = rows[0]
    base = next(r for r in rows if not r["tta"] and r["threshold"] == 0.5 and r["min_lesion_mm"] == 0)
    out = ROOT / "docs" / "results"
    (out / "postprocessing_validation.json").write_text(json.dumps({"validation_patients": VAL, "baseline": base,
                                                                    "chosen": best, "grid": rows}, indent=1))
    meta_p = ROOT / "models" / "model_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["postprocessing"] = {"tumour_threshold": best["threshold"], "min_lesion_mm": best["min_lesion_mm"],
                              "tta": best["tta"], "selected_on": f"validation patients {VAL}"}
    meta_p.write_text(json.dumps(meta, indent=1))
    fmt = lambda r: (f"tta={r['tta']!s:5} thr={r['threshold']} min={r['min_lesion_mm']:>2} mm | Dice/patient "
                     f"{r['dice_per_patient']:.3f} pooled {r['dice_pooled']:.3f} | recall {r['lesion_recall']:.2f} "
                     f"precision {r['lesion_precision']:.2f}")
    print("baseline:", fmt(base))
    print("top 5:"); [print("  " + fmt(r)) for r in rows[:5]]
    print("chosen  :", fmt(best))


if __name__ == "__main__":
    main()
