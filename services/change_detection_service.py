import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import ee

from config import CLASS_PALETTE, CLASS_LABELS
import services.ee_runtime as ee_runtime


DYNAMIC_WORLD_COLLECTION_ID = "GOOGLE/DYNAMICWORLD/V1"
PIXEL_ANALYSIS_SCALE_M = 100
DEFAULT_REGION = [54.16, 24.29, 54.74, 24.61]  # minLon, minLat, maxLon, maxLat


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
    if not ee_runtime.EE_READY:
        raise RuntimeError(f"Earth Engine is not ready: {ee_runtime.EE_ERROR}")

    aoi, region_label = parse_region(region_bbox, region_name)
    d1 = _parse_iso_date_dt(date1)
    d2 = _parse_iso_date_dt(date2)
    if d2 <= d1:
        raise ValueError("date2 must be after date1")
    if window_days < 1 or window_days > 180:
        raise ValueError("window_days must be between 1 and 180")

    dw1 = _build_dw_label_image(aoi, date1, window_days)
    dw2 = _build_dw_label_image(aoi, date2, window_days)

    # Match ChangeAnalysis.py exactly (band naming + thumbnail size)
    change_mask = dw1.neq(dw2).selfMask()
    paired = dw1.multiply(100).add(dw2).rename("pair")
    region_info = aoi.getInfo()
    leaflet_bounds = leaflet_bounds_from_geometry_info(region_info)

    before_url = dw1.getThumbURL(
        {
            "min": 0,
            "max": 8,
            "palette": CLASS_PALETTE,
            "dimensions": 512,
            "region": region_info,
        }
    )
    after_url = dw2.getThumbURL(
        {
            "min": 0,
            "max": 8,
            "palette": CLASS_PALETTE,
            "dimensions": 512,
            "region": region_info,
        }
    )
    change_url = change_mask.getThumbURL(
        {
            "min": 0,
            "max": 1,
            "palette": ["000000", "ff00ff"],
            "dimensions": 512,
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
