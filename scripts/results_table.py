"""Write the README results section from docs/results/test_metrics.json (so numbers never drift).

    python scripts/results_table.py          # replaces the block between the RESULTS markers in README.md
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "docs" / "results" / "test_metrics.json").read_text())


def pm(d):
    return f"{d['mean']:.3f} ± {d['std']:.3f}"


n = len(m["test_volumes"])
nt = m["tumour_dice_per_volume"]["n"]
lc = m["lesion_counts"]
block = f"""<!-- RESULTS:START -->
Held-out **test set: {n} LiTS patients** (seed-42 patient-wise split; never used for training or checkpoint selection), evaluated in 3-D at 512 × 512 with the exported ONNX models on CPU ([`scripts/evaluate.py`](scripts/evaluate.py), raw numbers in [`docs/results/test_metrics.json`](docs/results/test_metrics.json)).

| | Dice per patient (mean ± sd) | IoU per patient (mean ± sd) | Dice pooled over all test voxels | IoU pooled |
|---|---|---|---|---|
| **Liver** | {pm(m['liver_dice_per_volume'])} | {pm(m['liver_iou_per_volume'])} | {m['liver_dice_global']:.3f} | {m['liver_iou_global']:.3f} |
| **Tumour** ({nt} patients with tumour) | {pm(m['tumour_dice_per_volume'])} | {pm(m['tumour_iou_per_volume'])} | {m['tumour_dice_global']:.3f} | {m['tumour_iou_global']:.3f} |

| Lesion detection (3-D lesions) | Value |
|---|---|
| Recall, all lesions | {m['lesion_recall']:.1%} ({lc['det']} of {lc['gt']}) |
| Recall, lesions ≥ 10 mm | {m['lesion_recall_ge10mm']:.1%} ({lc['det10']} of {lc['gt10']}) |
| Precision (predicted lesions that overlap a real one) | {m['lesion_precision']:.1%} ({lc['pred_tp']} of {lc['pred']}) |

Validation during training (10 other patients, 2-D slices): liver Dice 0.929 (best epoch 12 of 14), tumour Dice 0.751 on liver crops (best epoch 16). Training curves and settings are in the notebook.

**How to read this.** Liver segmentation is close to the published range for LiTS. Tumour Dice per patient is lower and varies a lot between patients: big lesions are segmented well, while patients with only a few tiny lesions pull the mean down (a missed 5 mm lesion gives that patient a Dice near 0). The pooled Dice weighs every voxel equally and is dominated by large tumours. For reference, top 3-D methods on the LiTS leaderboard reach about 0.96 liver and 0.80 tumour Dice per case; this is a 2.5-D model trained in about two hours on a free GPU.

**The original model.** The SegNet shipped with the first version of this project produced empty masks on every liver slice we tried, through the original app and TensorFlow (see [docs/AUDIT.md](docs/AUDIT.md)); its notebook's 0.98 "Dice" was a per-pixel average dominated by background. The CV figure of 0.9 IoU is not supported by that notebook; the numbers above are what this version actually achieves.
<!-- RESULTS:END -->"""

readme = (ROOT / "README.md").read_text(encoding="utf-8")
if "<!-- RESULTS:START -->" in readme:
    readme = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, readme, flags=re.S)
else:
    readme = readme.replace("RESULTS_PLACEHOLDER", block)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print(block)
