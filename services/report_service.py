import json
import os
from typing import Any, Dict, List

from services.change_detection_service import _compute_landcover_metrics


OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()


def _fallback_narrative(gpt_in: dict) -> Dict[str, Any]:
    r = str(gpt_in.get("risk", "LOW")).upper()
    region = (str(gpt_in.get("region") or "").strip() or "This area")
    try:
        change = float(gpt_in.get("change") or 0)
    except (TypeError, ValueError):
        change = 0.0
    try:
        wl = float(gpt_in.get("water_loss") or 0)
    except (TypeError, ValueError):
        wl = 0.0
    try:
        vl = float(gpt_in.get("vegetation_loss") or 0)
    except (TypeError, ValueError):
        vl = 0.0
    try:
        bc = float(gpt_in.get("built_change") or 0)
    except (TypeError, ValueError):
        bc = 0.0
    tops: List[Dict[str, Any]] = list(gpt_in.get("top_transitions") or [])

    trans_bits: List[str] = []
    for t in tops[:3]:
        fn, tn = t.get("from"), t.get("to")
        pct = t.get("percent")
        if fn and tn and pct is not None:
            trans_bits.append(f"{fn} -> {tn} ({pct}%)")
    trans_sentence = (
        " Dominant label shifts include: " + "; ".join(trans_bits) + "."
        if trans_bits
        else " See the transition list below for where labels moved."
    )
    what_changed = (
        f"In {region}, about {change:.2f}% of pixels show a different Dynamic World class "
        f"between your two dates.{trans_sentence}"
    )

    risk_meaning = (
        f"The app assigned {r} using this run's signals: overall relabeling {change:.2f}% of the AOI, "
        f"net water surface loss {wl:.2f}%, net vegetation loss {vl:.2f}%, "
        f"and built-area change {bc:.2f}%. "
    )
    if r == "LOW":
        risk_meaning += (
            "Those stayed in the milder band for this simple score - useful as a screening flag, "
            "not a field survey."
        )
    elif r == "MEDIUM":
        risk_meaning += (
            "Several signals are elevated enough that checking timing (season, clouds) and "
            "ground context is worthwhile."
        )
    else:
        risk_meaning += (
            "Strong movement on change, water, vegetation, or built cover means you should "
            "cross-check against events, imagery quality, and local knowledge."
        )

    recs: List[str] = []
    if tops:
        t0 = tops[0]
        recs.append(
            f"Prioritize ground truth or high-res imagery for the top transition "
            f"({t0.get('from')} -> {t0.get('to')}, ~{t0.get('percent')}% of the AOI)."
        )
    if wl >= 0.5:
        recs.append(
            f"If the {wl:.2f}% net water loss looks wrong, verify date windows, clouds, "
            f"and seasonal water levels for {region}."
        )
    if vl >= 0.5:
        recs.append(
            f"Vegetation net loss is {vl:.2f}% - rule out harvest, drought, or misclassification "
            "before treating it as real loss."
        )
    if bc >= 1.0:
        recs.append(
            f"Built-area shift is {bc:.2f}% - compare with known construction or "
            "infrastructure projects for the same period."
        )
    recs.append(
        "Re-run with tighter date windows or a smaller bbox if noise dominates the signal."
    )
    recs.append("Export the figures and transition table for your report; note the exact dates used.")

    return {
        "what_changed": what_changed,
        "risk_meaning": risk_meaning,
        "recommendations": recs[:5],
    }


def _call_openai_narrative_only(gpt_in: dict) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return _fallback_narrative(gpt_in)
    try:
        from openai import OpenAI
    except ImportError:
        return _fallback_narrative(gpt_in)

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""You are an environmental analysis assistant.
Explain the results in a simple, user-friendly way for a general audience (not technical).

Data (already computed - do not recalculate risk or percentages):
{json.dumps(gpt_in, indent=2)}

Return ONLY valid JSON (no markdown fences) with exactly these keys:
- "what_changed": one short paragraph (2-4 sentences)
- "risk_meaning": one short paragraph explaining what the risk level means in plain English
- "recommendations": array of 3-5 short practical bullet strings

Do not mention hazard, exposure, vulnerability, or formulas."""

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = r.choices[0].message.content or "{}"
        out = json.loads(raw)
        for k in ("what_changed", "risk_meaning", "recommendations"):
            if k not in out:
                return _fallback_narrative(gpt_in)
        if not isinstance(out.get("recommendations"), list):
            return _fallback_narrative(gpt_in)
        return out
    except Exception:
        return _fallback_narrative(gpt_in)


def build_structured_report(payload: dict) -> Dict[str, Any]:
    stats = payload.get("change_stats") or {}
    region = payload.get("region") or stats.get("region_label") or "Study area"
    dr = payload.get("date_range") or {}
    start = dr.get("start", stats.get("before_date", ""))
    end = dr.get("end", stats.get("after_date", ""))

    change = float(stats.get("change_percent", 0) or 0)
    cb = stats.get("class_distribution_before") or []
    ca = stats.get("class_distribution_after") or []
    m = _compute_landcover_metrics(cb, ca, change)

    transitions = stats.get("top_transitions") or []
    top_for_gpt = [
        {
            "from": t.get("from_name"),
            "to": t.get("to_name"),
            "percent": t.get("percent_of_aoi"),
        }
        for t in transitions[:8]
    ]

    gpt_in = {
        "region": region,
        "change": round(change, 2),
        "risk": m["risk_level"],
        "water_loss": m["water_loss_percent"],
        "vegetation_loss": m["vegetation_loss_percent"],
        "built_change": m["built_change_percent"],
        "top_transitions": top_for_gpt,
    }

    narrative = _call_openai_narrative_only(gpt_in)

    return {
        "metrics": {
            "region": region,
            "date_range": {"start": start, "end": end},
            "change_percent": round(change, 2),
            "risk_level": m["risk_level"],
            "report_score": m["report_score"],
            "water_loss_percent": m["water_loss_percent"],
            "vegetation_loss_percent": m["vegetation_loss_percent"],
            "built_change_percent": m["built_change_percent"],
        },
        "top_transitions": transitions,
        "narrative": narrative,
        "gpt_used": bool(OPENAI_API_KEY and "PASTE_YOUR" not in OPENAI_API_KEY),
    }
