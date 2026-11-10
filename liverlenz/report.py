"""Structured report + scan-aware chat for the liver/tumour cascade.

Findings are always computed from the segmentation (deterministic). When an LLM is configured it
writes the Impression and answers chat questions; otherwise templated text is used, so the app
works with no API key.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from . import llm

DISCLAIMER = (
    "Research demo, not for clinical use. LiverLenz segments the liver and liver lesions on CT with "
    "a model trained on the public LiTS dataset. It can miss lesions (especially small ones) and can "
    "mark non-tumour tissue. It has not been clinically validated; every finding must be reviewed by "
    "a qualified radiologist."
)

SYSTEM_PROMPT = """You are LiverLenz, an assistant inside a research demo that segments the liver and liver tumours on CT.
Facts you must respect:
- A two-stage U-Net (trained on the public LiTS dataset) produced a liver mask and a tumour mask. You only know the numbers given below; you cannot see the images.
- The model does not classify lesions (it cannot tell HCC from metastasis, cyst or haemangioma) and can miss small lesions or produce false positives. Never state a diagnosis.
- Sizes are in mm/cm only when pixel spacing is known; otherwise they are in pixels.
- This is a research demo, not a medical device. Mention radiologist review briefly when clinical questions come up, without lecturing.
Be concise (under 150 words unless asked for more), technical and precise."""


def _fmt_size(l: dict) -> str:
    if l.get("max_diameter_mm"):
        s = f"{l['max_diameter_mm'] / 10:.1f} cm max diameter"
        if l.get("area_mm2"):
            s += f", {l['area_mm2'] / 100:.1f} cm² on its largest slice"
        if l.get("volume_ml") is not None:
            s += ", <0.1 mL" if l["volume_ml"] < 0.1 else f", ~{l['volume_ml']:.1f} mL"
        return s
    return f"{l['area_px']:,} px on its largest slice"


def _where(l: dict, w: int, h: int) -> str:
    cx, cy = l["centroid"]
    horiz = ["image-left (patient's right lobe side)", "central", "image-right (patient's left lobe side)"][min(int(3 * cx / w), 2)]
    vert = ["anterior", "mid", "posterior"][min(int(3 * cy / h), 2)]
    return f"{vert}, {horiz}"


def summary_text(s: dict | None) -> str:
    if not s:
        return "No scan has been analysed yet."
    w, h = s["original_size"]
    parts = [f"Input: {s['kind']} '{s['name']}', {s['n_slices']} slice(s), {w}x{h} px."]
    if s.get("spacing_mm"):
        parts.append(f"Pixel spacing {s['spacing_mm'][1]:.2f} mm" + (f", slice spacing {s['spacing_mm'][0]:.2f} mm." if s["n_slices"] > 1 else "."))
    else:
        parts.append("Pixel spacing unknown, so sizes are in pixels.")
    if not s["liver_detected"]:
        parts.append("No liver was segmented.")
        return " ".join(parts)
    if s.get("liver_volume_ml"):
        parts.append(f"Segmented liver volume ~{s['liver_volume_ml']:.0f} mL.")
    if not s["tumour_detected"]:
        parts.append("No tumour was segmented.")
    else:
        parts.append(f"{s['n_lesions']} lesion(s) segmented; tumour burden {s['tumour_burden_pct']:.1f}% of segmented liver"
                     + (f", total tumour volume ~{s['tumour_volume_ml']:.1f} mL." if s.get("tumour_volume_ml") else "."))
        for l in s["lesions"][:10]:
            x, y, bw, bh = l["bbox"]
            sl = f"slices {l['slices'][0]}-{l['slices'][1]}, " if s["n_slices"] > 1 else ""
            parts.append(f"Lesion {l['id']}: {_fmt_size(l)}; {sl}{_where(l, w, h)}; box x={x}, y={y}, {bw}x{bh} px; "
                         f"mean probability {l['mean_prob']:.2f}.")
    return " ".join(parts)


# ------------------------------------------------------------------ report
def _findings(s: dict) -> list[str]:
    w, h = s["original_size"]
    f = []
    if not s["liver_detected"]:
        return ["No liver was segmented. The image may not contain the liver, or it may not be an axial abdominal CT."]
    f.append("Liver segmented" + (f", volume ~{s['liver_volume_ml']:.0f} mL." if s.get("liver_volume_ml") else "."))
    if not s["tumour_detected"]:
        f.append("No focal liver lesion was segmented by the model.")
    else:
        f.append(f"{s['n_lesions']} lesion(s) segmented, tumour burden {s['tumour_burden_pct']:.1f}% of the segmented liver"
                 + (f" (~{s['tumour_volume_ml']:.1f} mL)." if s.get("tumour_volume_ml") else "."))
        for l in s["lesions"][:8]:
            x, y, bw, bh = l["bbox"]
            loc = f"slices {l['slices'][0]}–{l['slices'][1]}, " if s["n_slices"] > 1 else ""
            f.append(f"Lesion {l['id']}: {_fmt_size(l)}; {loc}{_where(l, w, h)}; bounding box x={x}, y={y}, {bw}×{bh} px.")
        if s["n_lesions"] > 8:
            f.append(f"{s['n_lesions'] - 8} further smaller lesion(s) not listed.")
    if not s.get("spacing_mm"):
        f.append("Pixel spacing unknown: sizes are given in pixels. Upload DICOM or NIfTI, or enter the spacing, for mm.")
    return f


def _template_impression(s: dict) -> str:
    if not s["liver_detected"]:
        return "No liver identified by the model; no assessment possible on this input."
    if not s["tumour_detected"]:
        return ("Liver segmented without any model-detected focal lesion. Small or low-contrast lesions can be missed; "
                "this does not exclude pathology.")
    big = s["lesions"][0]
    return (f"{s['n_lesions']} focal liver lesion(s) segmented, the largest measuring {_fmt_size(big)}. "
            "The model does not characterise lesions; correlation with contrast phases and radiologist review is required.")


def _llm_impression(s: dict) -> str | None:
    return llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Scan statistics: " + summary_text(s) + "\n\nWrite the 'Impression' of a short "
         "structured CT report: 2-3 sentences, plain text, no heading, no diagnosis or lesion type."},
    ], temperature=0.2, max_tokens=220)


def build_report(s: dict) -> dict:
    findings = _findings(s)
    impression = _llm_impression(s)
    source = "llm" if impression else "template"
    impression = impression or _template_impression(s)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w, h = s["original_size"]
    md = [
        "# LiverLenz AI: Liver Lesion Report", "",
        f"- **Generated:** {created}",
        f"- **Input:** {s['name']} ({s['kind']}, {s['n_slices']} slice(s), {w} × {h} px)",
        "- **Model:** two-stage U-Net (ResNet-18 encoder), trained on LiTS",
        f"- **Impression source:** {'LLM (' + (llm.status()['model'] or '') + ')' if source == 'llm' else 'template (LLM offline)'}",
        "", "## Findings", *[f"- {x}" for x in findings], "",
    ]
    if s["lesions"]:
        md += ["| # | Size | Slices | Bounding box (x, y, w, h) | Mean prob. |", "|---|---|---|---|---|"]
        for l in s["lesions"][:20]:
            md.append(f"| {l['id']} | {_fmt_size(l)} | {l['slices'][0]}–{l['slices'][1]} | {', '.join(map(str, l['bbox']))} | {l['mean_prob']:.2f} |")
        md.append("")
    md += ["## Impression", impression, "", "## Disclaimer", f"_{DISCLAIMER}_", ""]
    return {"findings": findings, "impression": impression, "disclaimer": DISCLAIMER,
            "source": source, "created": created, "markdown": "\n".join(md)}


# -------------------------------------------------------------------- chat
def _fallback(q: str, s: dict | None) -> str:
    q = q.lower()

    def has(*w):
        return any(re.search(r"\b" + x, q) for x in w)

    if has("hcc", "metasta", "malignan", "benign", "cancer", "cyst", "haemangioma", "hemangioma", "type", "diagnos"):
        return ("The model segments lesions but **does not classify them**: it cannot tell HCC from a metastasis, cyst or "
                "haemangioma. Characterisation needs multiphase imaging and a radiologist.")
    if s is None:
        if has("model", "how", "work", "accura", "dice", "iou", "train", "data"):
            return ("LiverLenz runs two U-Nets (ResNet-18 encoders) trained on the LiTS dataset: one finds the liver on the "
                    "whole slice, the second looks for tumours inside a crop around the liver. Accuracy on held-out test "
                    "patients is in the README results table.")
        return "Load a CT scan or pick an example first, then ask me about the lesions."
    if not s["liver_detected"]:
        return "No liver was found on this input, so there are no lesions to describe."
    if has("how many", "number", "count", "lesion", "tumou?r", "mass", "nodule", "find", "detect", "any"):
        if not s["tumour_detected"]:
            return "The model segmented **no tumour** on this input. Small or low-contrast lesions can be missed."
        big = s["lesions"][0]
        return (f"The model found **{s['n_lesions']} lesion(s)**, tumour burden **{s['tumour_burden_pct']:.1f}%** of the liver. "
                f"The largest is {_fmt_size(big)}.")
    if has("size", "big", "large", "diameter", "area", "volume", "cm", "mm"):
        if not s["tumour_detected"]:
            return "No lesion was segmented, so there is nothing to measure." + (
                f" The liver volume is ~{s['liver_volume_ml']:.0f} mL." if s.get("liver_volume_ml") else "")
        return " ".join(f"Lesion {l['id']}: {_fmt_size(l)}." for l in s["lesions"][:6])
    if has("where", "locat", "box", "bound", "position", "slice", "lobe"):
        if not s["tumour_detected"]:
            return "No lesion was segmented."
        w, h = s["original_size"]
        return " ".join(f"Lesion {l['id']}: {_where(l, w, h)}, box x={l['bbox'][0]}, y={l['bbox'][1]}, {l['bbox'][2]}×{l['bbox'][3]} px"
                        + (f", slices {l['slices'][0]}–{l['slices'][1]}" if s['n_slices'] > 1 else "") + "." for l in s["lesions"][:6])
    if has("confiden", "sure", "reliab", "trust", "accura", "dice", "iou"):
        probs = ", ".join(f"#{l['id']} {l['mean_prob']:.2f}" for l in s["lesions"][:6]) or "n/a"
        return (f"Mean tumour probability per lesion: {probs}. Held-out accuracy (per-patient Dice/IoU, lesion recall) "
                "is in the README; small lesions under 1 cm are the most often missed.")
    return summary_text(s) + " (LLM offline: answers are templated. Add an API key in `.env` for free-form chat.)"


def answer(question: str, s: dict | None, history: list[dict] | None = None) -> dict:
    s = s if isinstance(s, dict) and {"liver_detected", "tumour_detected", "lesions", "original_size", "n_slices",
                                       "kind", "name"} <= set(s) else None
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCurrent scan: " + summary_text(s)}]
    for t in (history or [])[-8:]:
        if t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str):
            msgs.append({"role": t["role"], "content": t["content"][:2000]})
    msgs.append({"role": "user", "content": question[:2000]})
    reply = llm.chat(msgs)
    if reply:
        return {"reply": reply, "source": "llm"}
    try:
        return {"reply": _fallback(question, s), "source": "template"}
    except (KeyError, TypeError, IndexError, ValueError):
        return {"reply": _fallback(question, None), "source": "template"}
