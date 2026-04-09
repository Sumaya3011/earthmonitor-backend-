# gee_utils.py
"""
Google Earth Engine / Dynamic World logic.

Builds per-day label mosaics and XYZ tile URLs for Leaflet.
"""

from datetime import date, timedelta

import ee

from config import CLASS_PALETTE, DW_MIN_DATE

# Full-world bounds (avoid poles where DW is sparse)
WORLD_GEOM = ee.Geometry.Rectangle([-179.99, -58.0, 179.99, 85.0])


def _clamp_date(d: date, max_d: date | None = None) -> date:
    max_d = max_d or date.today()
    if d < DW_MIN_DATE:
        return DW_MIN_DATE
    if d > max_d:
        return max_d
    return d


def build_dw_label_for_day(geom: ee.Geometry, day: date):
    """
    Per-pixel mode of Dynamic World ``label`` for one calendar day (UTC) inside ``geom``.
    """
    day = _clamp_date(day)
    start = day.isoformat()
    end = (day + timedelta(days=1)).isoformat()
    dw_image = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(geom)
        .select("label")
        .mode()
    )
    vis_params = {
        "min": 0,
        "max": 8,
        "palette": CLASS_PALETTE,
    }
    return dw_image, vis_params


def _image_to_tile_url(image: ee.Image, vis_params: dict) -> str | None:
    try:
        map_id = image.getMapId(vis_params)
        return map_id["tile_fetcher"].url_format
    except Exception as e:
        print("Error creating tile URL:", e)
        return None


def tile_url_for_day(geom: ee.Geometry, day: date) -> str | None:
    img, vis = build_dw_label_for_day(geom, day)
    return _image_to_tile_url(img, vis)


def get_dw_tile_urls_for_geometry(geom: ee.Geometry, day_a: date, day_b: date) -> dict:
    """
    Tile URLs for label mosaics on day_a, day_b, and a binary change layer.
    """
    day_a = _clamp_date(day_a)
    day_b = _clamp_date(day_b)

    img_a, vis = build_dw_label_for_day(geom, day_a)
    url_a = _image_to_tile_url(img_a, vis)

    img_b, _ = build_dw_label_for_day(geom, day_b)
    url_b = _image_to_tile_url(img_b, vis)

    change_img = img_a.neq(img_b)
    change_vis = {"min": 0, "max": 1, "palette": ["000000", "ff0000"]}
    url_change = _image_to_tile_url(change_img, change_vis)

    return {
        "a": url_a,
        "b": url_b,
        "change": url_change,
    }


def regional_geom(point: ee.Geometry, buffer_m: float = 55000.0) -> ee.Geometry:
    return point.buffer(buffer_m).bounds()
