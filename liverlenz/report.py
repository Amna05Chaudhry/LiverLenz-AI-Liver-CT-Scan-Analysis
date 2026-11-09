"""Structured report + scan-aware chat assistant.

Findings are always built from the measured mask statistics (deterministic).
When an LLM is configured it writes the Impression and answers chat questions;
otherwise templated text is used, so the app works with no API key.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from . import llm

DISCLAIMER = (
    "Research demo, not for clinical use. LiverLenz segments the liver region on a single "
    "2-D CT slice with a model trained on public data (LiTS). It does not detect or measure "
    "tumours and has not been clinically validated. All findings must be reviewed by a "
    "qualified radiologist."
)

SYSTEM_PROMPT = """You are LiverLenz, an assistant inside a research demo that segments the liver on single axial CT slices.
Facts you must respect:
- The segmentation model (a SegNet-style CNN, 128x128 input, trained on the public LiTS dataset) outputs a LIVER mask only. It does NOT detect, segment or characterise tumours or lesions. Never claim a tumour is present or absent.
- You only know the numeric statistics provided below; you cannot see the image itself.
- Measurements are 2-D, on one slice, in pixels unless a pixel spacing is given.
- This is a research demo, not a medical device. Encourage review by a radiologist when clinical questions come up, briefly and without lecturing.
Be concise (under 150 words unless asked for more), technical and precise. Use short paragraphs or bullet points."""


def stats_summary(stats: dict | None) -> str:
    if not stats:
        return "No scan has been analysed yet."
    if not stats.get("liver_detected"):
        return (f"Slice size {stats['image_width']}x{stats['image_height']} px. "
                "No liver tissue was segmented on this slice.")
    x, y, w, h = stats["bbox"]
    parts = [
        f"Slice size {stats['image_width']}x{stats['image_height']} px.",
        f"Segmented liver area: {stats['liver_area_px']:,} px ({stats['liver_area_pct']:.1f}% of the slice).",
        f"Bounding box: x={x}, y={y}, width={w}, height={h} px; centroid at ({stats['centroid'][0]}, {stats['centroid'][1]}).",
        f"Location: {stats['region']}.",
        f"Connected regions in mask: {stats['components']}.",
        f"Mean model confidence inside the mask: {stats['mean_confidence']:.2f}.",
    ]
    if stats.get("liver_area_mm2"):
        parts.append(f"With pixel spacing {stats['pixel_spacing_mm']} mm: area ~{stats['liver_area_mm2']:,} mm^2 "
                     f"({stats['liver_area_mm2'] / 100:.1f} cm^2), box {stats['bbox_mm'][0]} x {stats['bbox_mm'][1]} mm.")
    for note in stats.get("notes", []):
        parts.append(f"Note: {note}")
    return " ".join(parts)


# ------------------------------------------------------------------ report
def _findings(stats: dict) -> list[str]:
    if not stats.get("liver_detected"):
        return ["No liver tissue was segmented on this slice (mask is empty).",
                f"Slice dimensions: {stats['image_width']} x {stats['image_height']} px."]
    x, y, w, h = stats["bbox"]
    f = [
        f"Liver segmented in the {stats['region']}, covering {stats['liver_area_pct']:.1f}% of the slice "
        f"({stats['liver_area_px']:,} px).",
        f"Bounding box {w} x {h} px at (x={x}, y={y}); centroid ({stats['centroid'][0]}, {stats['centroid'][1]}).",
        f"Mask consists of {stats['components']} connected region{'s' if stats['components'] != 1 else ''}; "
        f"mean model confidence {stats['mean_confidence']:.2f}.",
    ]
    if stats.get("liver_area_mm2"):
        f.append(f"Using pixel spacing {stats['pixel_spacing_mm']} mm: area ~{stats['liver_area_mm2'] / 100:.1f} cm², "
                 f"extent {stats['bbox_mm'][0]} x {stats['bbox_mm'][1]} mm.")
    f.extend(stats.get("notes", []))
    return f


def _template_impression(stats: dict) -> str:
    if not stats.get("liver_detected"):
        return ("No liver was segmented. The slice may lie above or below the liver, or the image may not "
                "match the expected preprocessing (axial CT, liver window). Lesion assessment is outside the "
                "scope of this model.")
    pct = stats["liver_area_pct"]
    if pct >= 10:
        size = "a large liver cross-section, typical of a mid-hepatic level"
    elif pct >= 3:
        size = "a moderate liver cross-section"
    else:
        size = "a small liver cross-section, consistent with a slice near the dome or inferior tip"
    conf = stats["mean_confidence"]
    conf_txt = "high" if conf >= 0.85 else "moderate" if conf >= 0.7 else "low"
    frag = (" The mask is fragmented, so some regions may be false positives."
            if stats["components"] > 2 else "")
    return (f"Automated segmentation shows {size} in the {stats['region']}, with {conf_txt} model confidence."
            f"{frag} The liver mask is suitable for region-of-interest localisation; lesion assessment is "
            f"outside the scope of this model and requires radiologist review.")


def _llm_impression(stats: dict) -> str | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Scan statistics: " + stats_summary(stats) +
         "\n\nWrite the 'Impression' section of a short structured report: 2-3 sentences, plain text, "
         "no heading, no tumour claims, mention what the measurements suggest about slice level and segmentation quality."},
    ]
    return llm.chat(messages, temperature=0.2, max_tokens=220)


def build_report(stats: dict, filename: str | None = None) -> dict:
    findings = _findings(stats)
    impression = _llm_impression(stats)
    source = "llm" if impression else "template"
    impression = impression or _template_impression(stats)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = [
        "# LiverLenz AI: Segmentation Report",
        "",
        f"- **Generated:** {created}",
        f"- **Image:** {filename or 'uploaded slice'} ({stats['image_width']} x {stats['image_height']} px)",
        "- **Model:** SegNet-style CNN, 128x128 input, trained on LiTS",
        f"- **Impression source:** {'LLM (' + (llm.status()['model'] or '') + ')' if source == 'llm' else 'template (LLM offline)'}",
        "",
        "## Findings",
        *[f"- {f}" for f in findings],
        "",
        "## Impression",
        impression,
        "",
        "## Disclaimer",
        f"_{DISCLAIMER}_",
        "",
    ]
    return {"findings": findings, "impression": impression, "disclaimer": DISCLAIMER,
            "source": source, "created": created, "markdown": "\n".join(md)}


# -------------------------------------------------------------------- chat
def _fallback_answer(question: str, stats: dict | None) -> str:
    q = question.lower()
    has = bool(stats and stats.get("liver_detected"))

    def any_of(*words):
        return any(re.search(r"\b" + w, q) for w in words)

    if any_of("tumou?r", "lesion", "cancer", "malignan", "mass", "metasta", "hcc"):
        return ("This model segments the **liver only**; it was not trained to detect or measure tumours, so it "
                "cannot say whether a lesion is present. Lesion assessment needs a dedicated model and radiologist review.")
    if stats is None and not any_of("model", "how", "accura", "dice", "iou", "train", "data"):
        return "Upload a CT slice (or pick an example) first, then ask me about the segmentation."
    if any_of("area", "size", "big", "large", "small", "percent", "%", "volume"):
        if not has:
            return "No liver was segmented on this slice, so there is no area to report."
        extra = (f" With the pixel spacing you gave, that is about {stats['liver_area_mm2'] / 100:.1f} cm²."
                 if stats.get("liver_area_mm2") else " Add the pixel spacing (mm) to convert this to cm².")
        return (f"The segmented liver covers **{stats['liver_area_px']:,} px**, i.e. "
                f"**{stats['liver_area_pct']:.1f}%** of the slice.{extra} This is a 2-D cross-section, not organ volume.")
    if any_of("box", "bound", "where", "locat", "position", "region", "centroid", "side"):
        if not has:
            return "No liver was segmented, so there is no location to report."
        x, y, w, h = stats["bbox"]
        return (f"The liver lies in the **{stats['region']}**. Bounding box: x={x}, y={y}, {w} x {h} px; "
                f"centroid ({stats['centroid'][0]}, {stats['centroid'][1]}). In standard radiological display the "
                f"image left is the patient's right, where the liver normally sits.")
    if any_of("confiden", "sure", "reliab", "quality", "trust"):
        if not has:
            return "The model found no liver pixels above the 0.5 threshold on this slice."
        return (f"Mean model probability inside the mask is **{stats['mean_confidence']:.2f}** and the mask has "
                f"{stats['components']} connected region(s). " + " ".join(stats.get("notes", [])))
    if any_of("accura", "dice", "iou", "metric", "perform", "evaluat"):
        return ("On a held-out external dataset (CHAOS CT, never seen in training) the model's results are "
                "listed in the README's Results table. The notebook's 0.98 'Dice' is a per-pixel metric dominated by "
                "background and overstates performance.")
    if any_of("model", "how", "work", "architect", "train", "segnet", "data"):
        return ("LiverLenz uses a SegNet-style encoder-decoder CNN (4.1 M parameters) trained on 128x128 axial slices "
                "from the public LiTS dataset. Slices are converted to 3 channels (liver window, wider soft-tissue "
                "window, histogram-equalised), the network predicts a per-pixel liver probability, and pixels above 0.5 form the mask.")
    if has:
        return "Here is what I measured on this slice: " + stats_summary(stats) + \
               " (LLM offline: answers are templated. Add an API key in `.env` for free-form chat.)"
    return stats_summary(stats)


_REQUIRED = {"image_width": int, "image_height": int, "liver_detected": bool, "liver_area_px": int,
             "liver_area_pct": (int, float), "components": int}


def clean_stats(stats) -> dict | None:
    """Stats come back from the browser with chat requests: accept only well-formed ones."""
    if not isinstance(stats, dict):
        return None
    for k, t in _REQUIRED.items():
        if not isinstance(stats.get(k), t):
            return None
    if stats["liver_detected"]:
        for k in ("bbox", "centroid"):
            v = stats.get(k)
            if not (isinstance(v, list) and all(isinstance(i, int) for i in v)):
                return None
        if not isinstance(stats.get("mean_confidence"), (int, float)) or not isinstance(stats.get("region"), str):
            return None
    stats.setdefault("notes", [])
    if not isinstance(stats["notes"], list):
        stats["notes"] = []
    return stats


def answer(question: str, stats: dict | None, history: list[dict] | None = None) -> dict:
    stats = clean_stats(stats)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCurrent scan: " + stats_summary(stats)}]
    for turn in (history or [])[-8:]:
        if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
            messages.append({"role": turn["role"], "content": turn["content"][:2000]})
    messages.append({"role": "user", "content": question[:2000]})
    reply = llm.chat(messages)
    if reply:
        return {"reply": reply, "source": "llm"}
    return {"reply": _fallback_answer(question, stats), "source": "template"}
