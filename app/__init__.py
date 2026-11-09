"""LiverLenz AI web app (Flask)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from liverlenz import llm, report
from liverlenz.inference import (LiverSegmenter, draw_bbox, mask_image, overlay_mask,
                                 to_png_base64, to_rgb_array)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_SIDE = 1024

log = logging.getLogger("liverlenz")


def load_examples() -> list[dict]:
    meta = EXAMPLES_DIR / "examples.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return [{"file": p.name, "title": p.stem} for p in sorted(EXAMPLES_DIR.glob("*.jpg"))]


def create_app(segmenter: LiverSegmenter | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    seg = segmenter or LiverSegmenter()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify({"llm": llm.status(), "model": seg.model_path.name})

    @app.get("/api/examples")
    def examples():
        return jsonify(load_examples())

    @app.get("/examples/<path:name>")
    def example_file(name):
        return send_from_directory(EXAMPLES_DIR, name)

    @app.post("/api/analyze")
    def analyze():
        spacing = request.form.get("pixel_spacing", "").strip()
        try:
            spacing_mm = float(spacing) if spacing else None
            if spacing_mm is not None and not (0.05 <= spacing_mm <= 5):
                raise ValueError
        except ValueError:
            return jsonify(error="Pixel spacing must be a number between 0.05 and 5 mm."), 400

        example = request.form.get("example")
        if example:
            names = {e["file"] for e in load_examples()}
            if example not in names:
                return jsonify(error="Unknown example."), 400
            filename, source = example, EXAMPLES_DIR / example
        else:
            f = request.files.get("file")
            if not f or not f.filename:
                return jsonify(error="Please choose an image to analyse."), 400
            if Path(f.filename).suffix.lower() not in ALLOWED:
                return jsonify(error="Unsupported file type. Use PNG, JPG, BMP, TIFF or WEBP "
                                     "(convert DICOM/NIfTI with scripts/predict.py first)."), 400
            filename, source = Path(f.filename).name, f.stream

        try:
            img = Image.open(source)
            img.load()
        except (UnidentifiedImageError, OSError):
            return jsonify(error="That file could not be read as an image."), 400
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))

        t0 = time.perf_counter()
        try:
            result = seg.segment(to_rgb_array(img), pixel_spacing_mm=spacing_mm)
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            log.exception("Segmentation failed")
            return jsonify(error=f"Segmentation failed: {exc}"), 500
        elapsed = time.perf_counter() - t0

        stats = result.stats.to_dict()
        return jsonify({
            "filename": filename,
            "inference_ms": round(1000 * elapsed),
            "images": {
                "original": to_png_base64(result.image),
                "overlay": to_png_base64(overlay_mask(result.image, result.mask)),
                "bbox": to_png_base64(draw_bbox(overlay_mask(result.image, result.mask, 0.25), result.stats)),
                "mask": to_png_base64(mask_image(result.mask)),
            },
            "stats": stats,
            "report": report.build_report(stats, filename),
        })

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        if not message:
            return jsonify(error="Empty message."), 400
        stats = data.get("stats") if isinstance(data.get("stats"), dict) else None
        history = data.get("history") if isinstance(data.get("history"), list) else []
        return jsonify(report.answer(message, stats, history))

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File too large (max 10 MB)."), 413

    app.extensions["segmenter"] = seg
    return app
