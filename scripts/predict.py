"""Command-line liver segmentation (replaces the old main.py and jpeg.py).

Examples
--------
Segment one or more model-ready slices (3-channel windowed JPG/PNG)::

    python scripts/predict.py examples/chaos_p02_s050.jpg --out outputs

Convert a NIfTI CT volume to model-ready JPGs (what jpeg.py did) and segment them::

    python scripts/predict.py --nifti volume-33.nii --slices 40:120:10 --out outputs

Only export the JPGs, without running the model::

    python scripts/predict.py --nifti volume-33.nii --slices ::2 --export-only --out slices
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liverlenz.preprocess import hu_to_rgb, load_nifti  # noqa: E402


def parse_slice(spec: str) -> slice:
    parts = [int(p) if p else None for p in spec.split(":")]
    return slice(*parts) if len(parts) > 1 else slice(parts[0], parts[0] + 1)


def panel(images: list[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    gap = np.full((h, 8, 3), 255, np.uint8)
    out = []
    for i, im in enumerate(images):
        out.append(im)
        if i < len(images) - 1:
            out.append(gap)
    return np.concatenate(out, axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="model-ready slice images (JPG/PNG)")
    ap.add_argument("--nifti", help="CT volume (.nii / .nii.gz) in Hounsfield units")
    ap.add_argument("--slices", default="::2", help="slice range for --nifti, python syntax start:stop:step (default ::2)")
    ap.add_argument("--export-only", action="store_true", help="with --nifti: only save model-ready JPGs")
    ap.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    ap.add_argument("--spacing", type=float, help="pixel spacing in mm, to report areas in mm^2")
    ap.add_argument("--model", help="path to the .h5 model (default: models/liver_segmentation_model.h5)")
    ap.add_argument("--report", action="store_true", help="also write a Markdown report per slice")
    args = ap.parse_args(argv)

    if not args.images and not args.nifti:
        ap.error("give image paths or --nifti")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inputs: list[tuple[str, np.ndarray]] = []
    for p in args.images:
        inputs.append((Path(p).stem, np.array(Image.open(p).convert("RGB"))))
    if args.nifti:
        vol = load_nifti(args.nifti)
        stem = Path(args.nifti).name.split(".")[0]
        idx = range(vol.shape[-1])[parse_slice(args.slices)]
        print(f"{args.nifti}: volume {vol.shape}, exporting {len(idx)} slices")
        for i in idx:
            rgb = hu_to_rgb(vol[..., i])
            name = f"{stem}_slice_{i:03d}"
            if args.export_only:
                Image.fromarray(rgb).save(out / f"{name}.jpg", quality=90)
            else:
                inputs.append((name, rgb))
        if args.export_only:
            print(f"Saved {len(idx)} JPGs to {out}/")
            return 0

    from liverlenz import report
    from liverlenz.inference import LiverSegmenter, draw_bbox, overlay_mask

    seg = LiverSegmenter(args.model)
    for name, rgb in inputs:
        res = seg.segment(rgb, pixel_spacing_mm=args.spacing)
        s = res.stats
        Image.fromarray(panel([rgb, overlay_mask(rgb, res.mask), draw_bbox(rgb, s)])).save(out / f"{name}_result.png")
        (out / f"{name}.json").write_text(json.dumps(s.to_dict(), indent=2))
        if args.report:
            (out / f"{name}_report.md").write_text(report.build_report(s.to_dict(), name)["markdown"], encoding="utf-8")
        msg = (f"liver {s.liver_area_pct:5.1f}% of slice, bbox {s.bbox}, {s.region}, conf {s.mean_confidence}"
               if s.liver_detected else "no liver segmented")
        print(f"{name}: {msg}")
    print(f"Results written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
