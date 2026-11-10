"""Tests: preprocessing, lesion statistics, report/chat fallback, and the Flask API with the real models."""
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ["LLM_API_KEY"] = ""  # force the offline/template path

from app import create_app  # noqa: E402
from liverlenz import report  # noqa: E402
from liverlenz.cascade import (BASE, CascadeResult, LiverTumourCascade, Scan, liver_box, load_image,  # noqa: E402
                               scan_summary, slice_stats, window_u8)

MODELS = LiverTumourCascade().available


def test_window_matches_training():
    np.testing.assert_array_equal(window_u8(np.array([-500, -100, 75, 250, 900])), [0, 0, 127, 255, 255])


def test_liver_box_is_square_and_inside():
    m = np.zeros((BASE, BASE), bool); m[100:180, 300:500] = True
    y0, x0, s = liver_box(m)
    assert 0 <= y0 and 0 <= x0 and y0 + s <= BASE and x0 + s <= BASE and s >= 200


def _fake(spacing=(2.5, 0.8, 0.8)):
    sl = np.zeros((5, BASE, BASE), np.uint8)
    liver = np.zeros((5, BASE, BASE), bool); liver[1:4, 150:350, 100:300] = True
    tum = np.zeros_like(liver); tum[2:4, 200:240, 150:190] = True
    scan = Scan(sl, spacing, "volume", "fake.nii.gz", (BASE, BASE))
    res = CascadeResult(liver, tum, tum.astype(np.float32) * 0.9)
    res.lesions = LiverTumourCascade()._lesions(res, scan)
    return scan, res


def test_lesion_stats_from_masks():
    scan, res = _fake()
    assert len(res.lesions) == 1
    l = res.lesions[0]
    assert l.bbox == [150, 200, 40, 40] and l.slices == [2, 3]
    assert l.max_diameter_mm == pytest.approx(np.hypot(39, 39) * 0.8, rel=0.05)
    assert l.volume_ml == pytest.approx(2 * 40 * 40 * 2.5 * 0.64 / 1000, rel=0.01)
    s = scan_summary(scan, res)
    assert s["n_lesions"] == 1 and s["tumour_detected"] and s["liver_volume_ml"] > 0
    assert slice_stats(scan, res, 2)["lesions_on_slice"]


def test_report_and_fallback_chat():
    scan, res = _fake()
    s = scan_summary(scan, res)
    r = report.build_report(s)
    assert r["source"] == "template" and "Lesion 1" in " ".join(r["findings"]) and "| 1 |" in r["markdown"]
    assert "does not classify" in report.answer("Is it HCC?", s)["reply"]
    assert "1 lesion" in report.answer("How many tumours?", s)["reply"]
    assert report.answer("hello", {"bogus": 1})["source"] == "template"   # malformed client data


def test_image_loader_resizes_and_keeps_size():
    scan = load_image(np.zeros((300, 400), np.uint8), "x.png", pixel_spacing_mm=0.7)
    assert scan.slices.shape == (1, BASE, BASE) and scan.original_size == (400, 300)


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_status_and_examples(client):
    d = client.get("/api/status").get_json()
    assert d["llm"]["online"] is False
    assert isinstance(client.get("/api/examples").get_json(), list)


def test_bad_uploads(client):
    assert client.post("/api/analyze", data={}).status_code in (400, 503)
    r = client.post("/api/analyze", data={"file": (io.BytesIO(b"nope"), "x.txt")}, content_type="multipart/form-data")
    assert r.status_code in (400, 503)


@pytest.mark.skipif(not MODELS, reason="model files not present")
def test_analyze_example_and_slice(client):
    ex = client.get("/api/examples").get_json()
    vol = next(e for e in ex if e["file"].endswith(".nii.gz"))
    d = client.post("/api/analyze", data={"example": vol["file"]}).get_json()
    assert d["summary"]["liver_detected"] and d["images"]["overlay"].startswith("data:image/png;base64,")
    assert "## Findings" in d["report"]["markdown"]
    s = client.get(f"/api/slice/{d['id']}/0").get_json()
    assert s["slice"] == 0 and "boxes" in s["images"]


@pytest.mark.skipif(not MODELS, reason="model files not present")
def test_analyze_png_upload(client):
    buf = io.BytesIO(); Image.fromarray(np.full((256, 256), 30, np.uint8)).save(buf, "PNG"); buf.seek(0)
    r = client.post("/api/analyze", data={"file": (buf, "blank.png")}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["summary"]["n_slices"] == 1
