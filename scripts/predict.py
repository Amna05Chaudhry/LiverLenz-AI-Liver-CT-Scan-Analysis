"""Command-line liver + tumour segmentation (replaces the old main.py and jpeg.py).

    python scripts/predict.py examples/lits_0.nii.gz --out outputs           # NIfTI volume
    python scripts/predict.py scan.dcm --out outputs                          # one DICOM slice
    python scripts/predict.py slice.png --spacing 0.78 --out outputs          # plain CT image

For each input it writes ``<name>_summary.json`` (lesions with bounding boxes and sizes), a
Markdown report, and PNG panels (original | liver + tumour overlay | lesion boxes) for the key
slice and every slice that contains a lesion (``--all-slices`` for every slice).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liverlenz import report  # noqa: E402
from liverlenz.cascade import LiverTumourCascade, load_any, render, scan_summary  # noqa: E402


def panel(imgs: dict) -> np.ndarray:
    parts = [imgs["original"], imgs["overlay"], imgs["boxes"]]
    gap = np.full((parts[0].shape[0], 8, 3), 255, np.uint8)
    return np.concatenate([parts[0], gap, parts[1], gap, parts[2]], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="NIfTI (.nii/.nii.gz), DICOM (.dcm) or image files")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--spacing", type=float, help="pixel spacing in mm for plain images")
    ap.add_argument("--all-slices", action="store_true", help="save a panel for every slice")
    ap.add_argument("--models", help="folder with liver_stage1.onnx and tumour_stage2.onnx (default: models/)")
    args = ap.parse_args(argv)

    model = LiverTumourCascade(args.models)
    if not model.available:
        print(f"Model files not found in {model.dir}. See the README.", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in map(Path, args.inputs):
        stem = p.name.replace(".nii.gz", "").rsplit(".", 1)[0]
        scan = load_any(p.read_bytes(), p.name, args.spacing)
        res = model.run(scan)
        s = scan_summary(scan, res)
        (out / f"{stem}_summary.json").write_text(json.dumps(s, indent=2))
        (out / f"{stem}_report.md").write_text(report.build_report(s)["markdown"], encoding="utf-8")
        ks = range(len(scan.slices)) if args.all_slices else sorted({s["key_slice"], *[l["slice"] for l in s["lesions"]]})
        for k in ks:
            Image.fromarray(panel(render(scan, res, k))).save(out / f"{stem}_slice{k:03d}.png")
        msg = (f"{s['n_lesions']} lesion(s), tumour burden {s['tumour_burden_pct']:.1f}%" if s["tumour_detected"]
               else ("liver found, no tumour" if s["liver_detected"] else "no liver found"))
        print(f"{p.name}: {len(scan.slices)} slice(s): {msg}")
        for l in s["lesions"][:10]:
            size = f"{l['max_diameter_mm']} mm" if l["max_diameter_mm"] else f"{l['area_px']} px"
            print(f"   lesion {l['id']}: {size}, slices {l['slices'][0]}-{l['slices'][1]}, box {l['bbox']}")
    print(f"Results written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
