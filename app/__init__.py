"""LiverLenz AI web app (Flask): liver + tumour segmentation with a report and chat assistant."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from liverlenz import llm, report
from liverlenz.cascade import LiverTumourCascade, load_any, render, scan_summary, slice_stats, to_png_base64

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
ALLOWED = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".nii", ".nii.gz")
MAX_SLICES = 400

log = logging.getLogger("liverlenz")


def load_examples() -> list[dict]:
    meta = EXAMPLES_DIR / "examples.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else []


class ResultCache:
    """Keeps the last few analysed scans in memory so the slice slider can fetch other slices."""

    def __init__(self, size: int = 2):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def put(self, scan, res) -> str:
        sid = uuid.uuid4().hex[:12]
        res.tumour_prob = None            # free memory; lesion stats are already computed
        with self.lock:
            self.items[sid] = (scan, res)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return sid

    def get(self, sid):
        with self.lock:
            return self.items.get(sid)


def slice_payload(scan, res, k: int) -> dict:
    imgs = render(scan, res, k)
    return {"slice": k, "images": {name: to_png_base64(img) for name, img in imgs.items()},
            "stats": slice_stats(scan, res, k)}


def create_app(cascade: LiverTumourCascade | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
    model = cascade or LiverTumourCascade()
    cache = ResultCache()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify({"llm": llm.status(), "models_available": model.available, "model": model.meta})

    @app.get("/api/examples")
    def examples():
        return jsonify(load_examples())

    @app.get("/examples/<path:name>")
    def example_file(name):
        return send_from_directory(EXAMPLES_DIR, name)

    @app.post("/api/analyze")
    def analyze():
        if not model.available:
            return jsonify(error="Model files not found in models/. See the README to download or train them."), 503
        spacing = request.form.get("pixel_spacing", "").strip()
        try:
            spacing_mm = float(spacing) if spacing else None
            if spacing_mm is not None and not (0.05 <= spacing_mm <= 5):
                raise ValueError
        except ValueError:
            return jsonify(error="Pixel spacing must be a number between 0.05 and 5 mm."), 400

        example = request.form.get("example")
        if example:
            if example not in {e["file"] for e in load_examples()}:
                return jsonify(error="Unknown example."), 400
            name, data = example, (EXAMPLES_DIR / example).read_bytes()
            if spacing_mm is None:
                spacing_mm = next((e.get("pixel_spacing") for e in load_examples() if e["file"] == example), None)
        else:
            f = request.files.get("file")
            if not f or not f.filename:
                return jsonify(error="Please choose a CT image, DICOM file or NIfTI volume."), 400
            name = Path(f.filename).name
            if not name.lower().endswith(ALLOWED):
                return jsonify(error="Unsupported file type. Use NIfTI (.nii/.nii.gz), DICOM (.dcm) or an image (PNG/JPG)."), 400
            data = f.read()

        try:
            scan = load_any(data, name, spacing_mm)
        except Exception as exc:
            log.warning("Could not read %s: %s", name, exc)
            return jsonify(error="That file could not be read as a CT image, DICOM or NIfTI volume."), 400
        if len(scan.slices) > MAX_SLICES:
            return jsonify(error=f"Volume has {len(scan.slices)} slices; the demo accepts up to {MAX_SLICES}."), 400

        t0 = time.perf_counter()
        try:
            res = model.run(scan)
        except Exception as exc:  # pragma: no cover
            log.exception("Segmentation failed")
            return jsonify(error=f"Segmentation failed: {exc}"), 500
        elapsed = time.perf_counter() - t0

        summary = scan_summary(scan, res)
        k = summary["key_slice"]
        payload = slice_payload(scan, res, k)
        sid = cache.put(scan, res)
        return jsonify({"id": sid, "filename": name, "inference_ms": round(1000 * elapsed),
                        "summary": summary, "report": report.build_report(summary), **payload})

    @app.get("/api/slice/<sid>/<int:k>")
    def get_slice(sid, k):
        item = cache.get(sid)
        if not item:
            return jsonify(error="This scan is no longer in memory; please analyse it again."), 404
        scan, res = item
        if not 0 <= k < len(scan.slices):
            return jsonify(error="Slice out of range."), 400
        return jsonify(slice_payload(scan, res, k))

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        if not message:
            return jsonify(error="Empty message."), 400
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else None
        history = data.get("history") if isinstance(data.get("history"), list) else []
        return jsonify(report.answer(message, summary, history))

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File too large (max 300 MB). Use a compressed .nii.gz."), 413

    app.extensions["cascade"] = model
    return app
