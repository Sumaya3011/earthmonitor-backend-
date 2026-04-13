# main.py
import json
import os
import io
import math
import tempfile
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import ee
import imageio.v2 as imageio
import numpy as np
import requests
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from pydantic import BaseModel

from config import (
    YEARS,
    LOCATION_LAT,
    LOCATION_LON,
    LOCATION_NAME,
    CLASS_PALETTE,
    CLASS_LABELS,
    DW_MIN_DATE,
)
from gee_utils import get_dw_tile_urls, tile_url_at_point, tile_url_global_year
from chat_utils import ask_chatbot

# ---------------------------------------------------------------------
# FastAPI app + static frontend
# ---------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
DYNAMIC_WORLD_COLLECTION_ID = "GOOGLE/DYNAMICWORLD/V1"
PIXEL_ANALYSIS_SCALE_M = 100
DEFAULT_REGION = [54.16, 24.29, 54.74, 24.61]  # minLon, minLat, maxLon, maxLat



@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ---------------------------------------------------------------------
# Earth Engine init
# ---------------------------------------------------------------------
EE_READY = False
EE_ERROR = None


def init_ee():
    global EE_READY, EE_ERROR
    if EE_READY:
        return

    try:
        service_account_json = (os.environ.get("EE_SERVICE_ACCOUNT_JSON") or "").strip()
        if not service_account_json:
            raise RuntimeError(
                "EE_SERVICE_ACCOUNT_JSON is missing. "
                "Set it in Render → Environment (full JSON key as one line or escaped)."
            )

        info = json.loads(service_account_json)
        email = info["client_email"]
        # Prefer EE_PROJECT if set (Render/GCP); else project_id inside the key JSON
        project_id = (os.environ.get("EE_PROJECT") or "").strip() or info.get("project_id")

        credentials = ee.ServiceAccountCredentials(email, key_data=service_account_json)
        if project_id:
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(credentials)

        EE_READY = True
        EE_ERROR = None
        print("Earth Engine initialized successfully.", "project=", project_id or "(default)")
    except Exception as e:
        EE_READY = False
        EE_ERROR = str(e)
        print("Failed to initialize Earth Engine:", EE_ERROR)


@app.on_event("startup")
async def startup_event():
    init_ee()


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class MapRequest(BaseModel):
    mode: str
    date_a: Optional[str] = None
    date_b: Optional[str] = None
    city: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    mode: str
    date_a: Optional[str] = None
    date_b: Optional[str] = None
    city: Optional[str] = None


class VideoRequest(BaseModel):
    year_a: int
    year_b: int
    city: Optional[str] = None
    fps: int = 2
    size: int = 768
    radius_m: int = 5000


class ChangeBody(BaseModel):
    date1: str
    date2: str
    region: Optional[str] = None
    region_name: Optional[str] = None
    window_days: int = 30


class ReportBody(BaseModel):
    region: str
    date_range: dict
    change_stats: dict


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
geolocator = Nominatim(user_agent="dw-change-app")


def parse_iso_date(s: Optional[str]) -> date:
    if not s or not str(s).strip():
        return date.today() - timedelta(days=2)
    try:
        return date.fromisoformat(str(s).strip()[:10])
    except ValueError:
        return date.today() - timedelta(days=2)


def clamp_map_date(d: date) -> date:
    today = date.today()
    if d < DW_MIN_DATE:
        return DW_MIN_DATE
    if d > today:
        return today
    return d


def display_date(d: date) -> str:
    return d.strftime("%d %b %Y")


def resolve_city(city: Optional[str]):
    if not city or not city.strip():
        return LOCATION_NAME, LOCATION_LAT, LOCATION_LON

    try:
        loc = geolocator.geocode(city.strip())
        if loc:
            return city.strip(), loc.latitude, loc.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass

    return LOCATION_NAME, LOCATION_LAT, LOCATION_LON


def month_sequence(year_a: int, year_b: int):
    months = []
    for y in range(year_a, year_b + 1):
        for m in range(1, 13):
            months.append((y, m))
    return months


def next_month(y: int, m: int):
    if m == 12:
        return y + 1, 1
    return y, m + 1


def monthly_dw_visual(region: ee.Geometry, y: int, m: int) -> ee.Image:
    start = f"{y:04d}-{m:02d}-01"
    ny, nm = next_month(y, m)
    end = f"{ny:04d}-{nm:02d}-01"

    palette = ["#" + c for c in CLASS_PALETTE]

    img = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .select("label")
        .mode()
    )

    return img.visualize(min=0, max=8, palette=palette).clip(region)


def add_frame_label(img: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    pad = 12
    box_h = 34
    draw.rounded_rectangle(
        [pad, pad, 220, pad + box_h],
        radius=10,
        fill=(15, 23, 42, 210),
        outline=(100, 116, 139, 255),
        width=1,
    )
    draw.text((pad + 10, pad + 9), label, fill=(229, 231, 235))
    return img


def ee_region_bbox(region: ee.Geometry):
    coords = region.bounds().coordinates().getInfo()[0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def _looks_like_bbox(region_str: str) -> bool:
    parts = [p.strip() for p in region_str.split(",")]
    if len(parts) != 4:
        return False
    try:
        [float(x) for x in parts]
        return True
    except ValueError:
        return False


def geocode_place(name: str) -> Optional[List[float]]:
    if not name or not name.strip():
        return None
    q = urllib.parse.quote(name.strip())
    url = (
        "https://nominatim.openstreetmap.org/search?"
        f"q={q}&format=json&limit=1&addressdetails=0"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "UAU-ChangeAnalysis/1.0 (education)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not data:
        return None
    bb = data[0].get("boundingbox")
    if not bb or len(bb) < 4:
        return None
    lat_s, lat_n = float(bb[0]), float(bb[1])
    lon_w, lon_e = float(bb[2]), float(bb[3])
    lat_pad = max((lat_n - lat_s) * 0.08, 0.02)
    lon_pad = max((lon_e - lon_w) * 0.08, 0.02)
    return [
        lon_w - lon_pad,
        lat_s - lat_pad,
        lon_e + lon_pad,
        lat_n + lat_pad,
    ]


def parse_region(
    region_bbox: Optional[str],
    region_name: Optional[str],
) -> Tuple[ee.Geometry, str]:
    if region_bbox and _looks_like_bbox(region_bbox):
        coords = [float(x) for x in region_bbox.split(",")]
        return ee.Geometry.Rectangle(coords), "Custom bounding box"
    if region_name and region_name.strip():
        g = geocode_place(region_name)
        if g:
            return ee.Geometry.Rectangle(g), region_name.strip()
    return ee.Geometry.Rectangle(DEFAULT_REGION), "Default AOI (Abu Dhabi block)"


def leaflet_bounds_from_geometry_info(geom_info: Optional[dict]) -> List[List[float]]:
    if not geom_info:
        return [[24.29, 54.16], [24.61, 54.74]]
    try:
        coords = geom_info["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [[min(lats), min(lons)], [max(lats), max(lons)]]
    except (KeyError, IndexError, TypeError):
        return [[24.29, 54.16], [24.61, 54.74]]


def _hist_to_class_rows(hist: Optional[dict], total: float) -> List[Dict[str, Any]]:
    if not hist or not isinstance(hist, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for k, v in hist.items():
        try:
            cid = int(float(k))
        except (ValueError, TypeError):
            continue
        if cid < 0 or cid > 8:
            continue
        c = float(v) if v is not None else 0.0
        pct = round((c / max(total, 1.0)) * 100, 2)
        rows.append(
            {
                "id": cid,
                "name": CLASS_LABELS[cid],
                "pixel_count": int(c),
                "percent": pct,
            }
        )
    rows.sort(key=lambda x: x["id"])
    return rows


def _parse_transition_rows(
    pair_hist: Optional[dict], total: float, limit: int = 15
) -> List[Dict[str, Any]]:
    if not pair_hist or not isinstance(pair_hist, dict):
        return []
    raw: List[Dict[str, Any]] = []
    for k, v in pair_hist.items():
        try:
            code = int(float(k))
        except (ValueError, TypeError):
            continue
        from_id = code // 100
        to_id = code % 100
        if from_id < 0 or from_id > 8 or to_id < 0 or to_id > 8:
            continue
        if from_id == to_id:
            continue
        c = float(v) if v is not None else 0.0
        raw.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "from_name": CLASS_LABELS[from_id],
                "to_name": CLASS_LABELS[to_id],
                "pixel_count": int(c),
                "percent_of_aoi": round((c / max(total, 1.0)) * 100, 2),
            }
        )
    raw.sort(key=lambda x: -x["pixel_count"])
    return raw[:limit]


def _parse_iso_date_dt(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _build_dw_label_image(aoi: ee.Geometry, center_date: str, window_days: int) -> ee.Image:
    center = _parse_iso_date_dt(center_date)
    start = (center - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=window_days)).strftime("%Y-%m-%d")
    return (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION_ID)
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("label")
        .mode()
        .clip(aoi)
    )


def _pct_for_class(rows: List[Dict[str, Any]], class_id: int) -> float:
    for r in rows:
        if int(r.get("id", -1)) == class_id:
            return float(r.get("percent", 0) or 0)
    return 0.0


def _vegetation_pct(rows: List[Dict[str, Any]]) -> float:
    return sum(_pct_for_class(rows, i) for i in (1, 2, 3, 4, 5))


def _compute_landcover_metrics(
    class_before: List[Dict[str, Any]],
    class_after: List[Dict[str, Any]],
    change_pct: float,
) -> Dict[str, Any]:
    w0 = _pct_for_class(class_before, 0)
    w1 = _pct_for_class(class_after, 0)
    water_loss = max(0.0, round(w0 - w1, 2))

    v0 = _vegetation_pct(class_before)
    v1 = _vegetation_pct(class_after)
    veg_loss = max(0.0, round(v0 - v1, 2))

    b0 = _pct_for_class(class_before, 6)
    b1 = _pct_for_class(class_after, 6)
    built_change = round(abs(b1 - b0), 2)

    ch = float(change_pct)
    score = 0
    if ch < 5:
        score += 1
    elif ch < 15:
        score += 2
    else:
        score += 3

    if water_loss < 1:
        score += 1
    elif water_loss < 3:
        score += 2
    else:
        score += 3

    if veg_loss < 1:
        score += 1
    elif veg_loss < 3:
        score += 2
    else:
        score += 3

    if built_change < 2:
        score += 1
    elif built_change < 5:
        score += 2
    else:
        score += 3

    if score <= 5:
        risk = "LOW"
    elif score <= 8:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    if ch > 20 or water_loss > 7:
        risk = "HIGH"

    return {
        "water_loss_percent": water_loss,
        "vegetation_loss_percent": veg_loss,
        "built_change_percent": built_change,
        "report_score": score,
        "risk_level": risk,
    }


def compute_change_detection(
    date1: str,
    date2: str,
    region_bbox: Optional[str] = None,
    region_name: Optional[str] = None,
    window_days: int = 30,
) -> Dict[str, Any]:
    if not EE_READY:
        raise RuntimeError(f"Earth Engine is not ready: {EE_ERROR}")

    aoi, region_label = parse_region(region_bbox, region_name)
    d1 = _parse_iso_date_dt(date1)
    d2 = _parse_iso_date_dt(date2)
    if d2 <= d1:
        raise ValueError("date2 must be after date1")
    if window_days < 1 or window_days > 180:
        raise ValueError("window_days must be between 1 and 180")

    dw1 = _build_dw_label_image(aoi, date1, window_days)
    dw2 = _build_dw_label_image(aoi, date2, window_days)

    change_mask = dw1.neq(dw2).selfMask().rename("label")
    paired = dw1.multiply(100).add(dw2).rename("pair")
    region_info = aoi.getInfo()
    leaflet_bounds = leaflet_bounds_from_geometry_info(region_info)

    before_url = dw1.getThumbURL(
        {
            "min": 0,
            "max": 8,
            "palette": CLASS_PALETTE,
            "dimensions": 768,
            "region": region_info,
        }
    )
    after_url = dw2.getThumbURL(
        {
            "min": 0,
            "max": 8,
            "palette": CLASS_PALETTE,
            "dimensions": 768,
            "region": region_info,
        }
    )
    change_url = change_mask.getThumbURL(
        {
            "min": 0,
            "max": 1,
            "palette": ["000000", "ff00ff"],
            "dimensions": 768,
            "region": region_info,
        }
    )

    class_before: List[Dict[str, Any]] = []
    class_after: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []

    scale = PIXEL_ANALYSIS_SCALE_M
    total_reduce = (
        dw1.mask(ee.Image(1))
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )
    total_pixels = int((total_reduce or {}).get("label") or 0)

    change_reduce = (
        change_mask.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )
    change_pixels = int((change_reduce or {}).get("label") or 0)
    change_pct = round((change_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0

    h1 = (
        dw1.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )
    h2 = (
        dw2.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )
    hist1 = h1.get("label") if h1 else None
    hist2 = h2.get("label") if h2 else None
    if isinstance(hist1, dict):
        t1 = sum(float(v) for v in hist1.values())
        class_before = _hist_to_class_rows(hist1, t1)
    if isinstance(hist2, dict):
        t2 = sum(float(v) for v in hist2.values())
        class_after = _hist_to_class_rows(hist2, t2)

    hp = (
        paired.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )
    pair_hist = hp.get("pair") if hp else None
    if isinstance(pair_hist, dict):
        transitions = _parse_transition_rows(pair_hist, float(total_pixels))

    metrics = _compute_landcover_metrics(class_before, class_after, change_pct)

    return {
        "before_url": before_url,
        "after_url": after_url,
        "change_url": change_url,
        "change_percent": change_pct,
        "before_date": date1,
        "after_date": date2,
        "window_days": window_days,
        "time_span_years": round((d2 - d1).days / 365.25, 2),
        "region_label": region_label,
        "leaflet_bounds": leaflet_bounds,
        "pixel_scale_m": PIXEL_ANALYSIS_SCALE_M,
        "total_sampled_pixels": total_pixels,
        "changed_pixels": change_pixels,
        "class_distribution_before": class_before,
        "class_distribution_after": class_after,
        "top_transitions": transitions,
        "risk_level": metrics["risk_level"],
        "report_score": metrics["report_score"],
        "water_loss_percent": metrics["water_loss_percent"],
        "vegetation_loss_percent": metrics["vegetation_loss_percent"],
        "built_change_percent": metrics["built_change_percent"],
        "dynamic_world_collection": DYNAMIC_WORLD_COLLECTION_ID,
        "dynamic_world_note": (
            f"Collection {DYNAMIC_WORLD_COLLECTION_ID}: compared with +/-{window_days} day windows at ~{scale} m."
        ),
    }


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
        "gpt_used": bool(OPENAI_API_KEY),
    }


def download_month_frame(region: ee.Geometry, y: int, m: int, size: int) -> np.ndarray:
    vis = monthly_dw_visual(region, y, m)
    bbox = ee_region_bbox(region)

    url = vis.getThumbURL({
        "region": bbox,
        "dimensions": size,
        "format": "png",
    })

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = add_frame_label(img, f"{y}-{m:02d}")
    return np.array(img)


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ee_ready": EE_READY,
        "ee_error": EE_ERROR,
    }


# ---------------------------------------------------------------------
# Map config
# ---------------------------------------------------------------------
@app.post("/map-config")
def map_config(req: MapRequest):
    init_ee()
    if not EE_READY:
        raise HTTPException(
            status_code=500,
            detail=f"Earth Engine is not ready: {EE_ERROR}",
        )

    mode = req.mode
    da = clamp_map_date(parse_iso_date(req.date_a))
    db = clamp_map_date(parse_iso_date(req.date_b))
    if db < da:
        da, db = db, da

    # Dynamic World uses annual composites (original app): year from each selected date
    year_a = da.year
    year_b = db.year
    if year_a not in YEARS:
        year_a = YEARS[0]
    if year_b not in YEARS:
        year_b = YEARS[-1]

    if mode == "home":
        if not req.city or not str(req.city).strip():
            url = tile_url_global_year(year_a)
            return {
                "city": "World",
                "center_lat": 15.0,
                "center_lon": 0.0,
                "date_a": da.isoformat(),
                "date_b": da.isoformat(),
                "date_a_display": display_date(da),
                "date_b_display": display_date(da),
                "dw_year_a": year_a,
                "dw_year_b": year_a,
                "mode": mode,
                "tiles": {"a": url, "b": None, "change": None},
                "map_zoom": 2,
            }

        city_name, lat, lon = resolve_city(req.city)
        point = ee.Geometry.Point([lon, lat])
        url = tile_url_at_point(point, year_a)
        return {
            "city": city_name,
            "center_lat": lat,
            "center_lon": lon,
            "date_a": da.isoformat(),
            "date_b": da.isoformat(),
            "date_a_display": display_date(da),
            "date_b_display": display_date(da),
            "dw_year_a": year_a,
            "dw_year_b": year_a,
            "mode": mode,
            "tiles": {"a": url, "b": None, "change": None},
            "map_zoom": 11,
        }

    city_name, lat, lon = resolve_city(req.city)
    point = ee.Geometry.Point([lon, lat])
    tiles = get_dw_tile_urls(point, year_a, year_b)

    return {
        "city": city_name,
        "center_lat": lat,
        "center_lon": lon,
        "date_a": da.isoformat(),
        "date_b": db.isoformat(),
        "date_a_display": display_date(da),
        "date_b_display": display_date(db),
        "dw_year_a": year_a,
        "dw_year_b": year_b,
        "mode": mode,
        "tiles": tiles,
    }


@app.post("/api/change-detection")
def api_change(body: ChangeBody):
    try:
        return compute_change_detection(
            body.date1,
            body.date2,
            region_bbox=body.region,
            region_name=body.region_name,
            window_days=body.window_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/report")
def api_report(body: ReportBody):
    try:
        return build_structured_report(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    city_name, lat, lon = resolve_city(req.city)
    da = clamp_map_date(parse_iso_date(req.date_a))
    db = clamp_map_date(parse_iso_date(req.date_b))

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that explains Dynamic World land cover "
            "maps and changes over time in SIMPLE language. "
            "The app uses dates to pick calendar years; Dynamic World layers are annual composites "
            "(mode of daily labels over each year), matching the standard Earth Engine recipe. "
            "Modes: home (single map; world view if no region, else zoomed region), "
            "change_detection (two maps, date A vs B), timeseries, prediction. "
            "In prediction mode the map shows the same historical A/B/change layers as time series; "
            "your job is to discuss plausible future land-cover outcomes qualitatively, "
            "not as a numerical forecast, unless the user asks for general education. "
            f"Current mode: {req.mode}, date A: {display_date(da)}, date B: {display_date(db)}. "
            f"Current region: {city_name} at ({lat:.3f}, {lon:.3f}). "
            "Return your answer STRICTLY as JSON with two keys: "
            "'explanation' and 'summary'. "
            "'explanation' can be a normal short paragraph. "
            "'summary' must be at most two short sentences."
        ),
    }

    messages_for_api = [
        system_msg,
        {"role": "user", "content": req.message},
    ]

    raw = ask_chatbot(messages_for_api)

    explanation = raw
    summary = raw

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            explanation = data.get("explanation", raw)
            summary = data.get("summary", explanation)
    except Exception:
        parts = explanation.split(".")
        summary = ".".join(parts[:2]).strip()
        if summary and not summary.endswith("."):
            summary += "."

    return {
        "reply": explanation,
        "summary": summary,
    }


# ---------------------------------------------------------------------
# Time series video
# ---------------------------------------------------------------------
@app.post("/timeseries-video")
def timeseries_video(req: VideoRequest):
    init_ee()
    if not EE_READY:
        raise HTTPException(
            status_code=500,
            detail=f"Earth Engine is not ready: {EE_ERROR}",
        )

    if req.year_a > req.year_b:
        raise HTTPException(status_code=400, detail="year_a must be <= year_b")

    city_name, lat, lon = resolve_city(req.city)

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(req.radius_m).bounds()

    months = month_sequence(req.year_a, req.year_b)
    frames = []

    for y, m in months:
        try:
            frame = download_month_frame(region, y, m, req.size)
            frames.append(frame)
        except Exception as e:
            print(f"Skipping frame {y}-{m:02d}: {e}")

    if not frames:
        raise HTTPException(status_code=500, detail="Could not generate any monthly frames.")

    safe_city = city_name.replace(" ", "_").replace("/", "_")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    writer = imageio.get_writer(
        tmp_path,
        fps=max(1, req.fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    filename = f"timeseries_{safe_city}_{req.year_a}_{req.year_b}.mp4"
    return FileResponse(
        tmp_path,
        media_type="video/mp4",
        filename=filename,
    )
