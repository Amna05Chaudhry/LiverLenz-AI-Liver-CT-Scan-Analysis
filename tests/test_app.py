"""Smoke tests: preprocessing, stats, report fallback, and the Flask API with the real model."""
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ["LLM_API_KEY"] = ""  # force the offline/template path

from app import create_app  # noqa: E402
from liverlenz import report  # noqa: E402
from liverlenz.inference import compute_stats  # noqa: E402
from liverlenz.preprocess import hu_to_rgb, windowed  # noqa: E402


def test_windowing_matches_fastai_definition():
    hu = np.array([-1000, -45, 30, 105, 2000], np.float32)
    np.testing.assert_allclose(windowed(hu, 150, 30), [0, 0, 0.5, 1, 1])


def test_hu_to_rgb_shape_and_range():
    hu = np.random.default_rng(0).normal(40, 200, (64, 64)).astype(np.float32)
    rgb = hu_to_rgb(hu)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8


def test_stats_on_synthetic_mask():
    mask = np.zeros((100, 200), np.uint8)
    mask[10:40, 20:80] = 1
    s = compute_stats(mask, mask.astype(np.float32), pixel_spacing_mm=0.5)
    assert s.liver_detected and s.liver_area_px == 30 * 60
    assert s.bbox == [20, 10, 60, 30]
    assert s.liver_area_mm2 == pytest.approx(1800 * 0.25)
    assert "upper left" in s.region


def test_empty_mask_report():
    s = compute_stats(np.zeros((50, 50), np.uint8), np.zeros((50, 50), np.float32)).to_dict()
    r = report.build_report(s, "x.png")
    assert r["source"] == "template" and "No liver" in r["findings"][0]
    assert "## Disclaimer" in r["markdown"]


def _liver_stats():
    mask = np.zeros((128, 128), np.uint8)
    mask[30:90, 10:70] = 1
    return compute_stats(mask, mask * 0.9).to_dict()


def test_fallback_chat_refuses_tumour_claims():
    out = report.answer("Is there a tumour?", _liver_stats())
    assert out["source"] == "template" and "liver only" in out["reply"]


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_status(client):
    d = client.get("/api/status").get_json()
    assert d["llm"]["online"] is False


def test_analyze_example(client):
    examples = client.get("/api/examples").get_json()
    assert 3 <= len(examples) <= 6
    liver = next(e for e in examples if e.get("expect_liver", True))
    d = client.post("/api/analyze", data={"example": liver["file"]}).get_json()
    assert d["stats"]["liver_detected"]
    assert d["images"]["overlay"].startswith("data:image/png;base64,")
    assert "## Findings" in d["report"]["markdown"]


def test_analyze_upload_and_errors(client):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(buf, "PNG")
    buf.seek(0)
    r = client.post("/api/analyze", data={"file": (buf, "black.png")}, content_type="multipart/form-data")
    assert r.status_code == 200
    r = client.post("/api/analyze", data={"file": (io.BytesIO(b"not an image"), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "error" in r.get_json()
    assert client.post("/api/analyze", data={}).status_code == 400


def test_chat_endpoint(client):
    stats = _liver_stats()
    d = client.post("/api/chat", json={"message": "How large is the liver?", "stats": stats}).get_json()
    assert f"{stats['liver_area_pct']:.1f}%" in d["reply"]
    # malformed client stats must not crash the endpoint
    r = client.post("/api/chat", json={"message": "Where is it?", "stats": {"liver_detected": True}})
    assert r.status_code == 200 and "reply" in r.get_json()
