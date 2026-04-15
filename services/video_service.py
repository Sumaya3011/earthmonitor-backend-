import io
import tempfile
from datetime import date, timedelta
from typing import Iterator, Optional, Tuple

import ee
import imageio.v2 as imageio
import numpy as np
import requests
from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw

from config import CLASS_PALETTE
from schemas.requests import VideoRequest
from services.ee_runtime import init_ee
import services.ee_runtime as ee_runtime
from services.map_service import resolve_city


def next_month(y: int, m: int):
    if m == 12:
        return y + 1, 1
    return y, m + 1


def dw_visual_for_date_range(region: ee.Geometry, start_iso: str, end_iso: str) -> ee.Image:
    palette = ["#" + c for c in CLASS_PALETTE]

    img = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start_iso, end_iso)
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


def download_dw_frame(region: ee.Geometry, start_iso: str, end_iso: str, size: int, label: str) -> np.ndarray:
    vis = dw_visual_for_date_range(region, start_iso, end_iso)
    bbox = ee_region_bbox(region)

    url = vis.getThumbURL({
        "region": bbox,
        "dimensions": size,
        "format": "png",
    })

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = add_frame_label(img, label)
    return np.array(img)


def download_month_frame(region: ee.Geometry, y: int, m: int, size: int) -> np.ndarray:
    start = f"{y:04d}-{m:02d}-01"
    ny, nm = next_month(y, m)
    end = f"{ny:04d}-{nm:02d}-01"
    return download_dw_frame(region, start, end, size, f"{y}-{m:02d}")


def parse_iso_date(s: str) -> Optional[date]:
    if not s or not str(s).strip():
        return None
    try:
        return date.fromisoformat(str(s).strip()[:10])
    except ValueError:
        return None


def resolve_video_date_range(req: VideoRequest) -> Tuple[date, date]:
    da = parse_iso_date(req.date_a or "")
    db = parse_iso_date(req.date_b or "")
    if da and db:
        if db < da:
            da, db = db, da
        return da, db
    return date(req.year_a, 1, 1), date(req.year_b, 12, 31)


def iter_months_inclusive(d0: date, d1: date) -> Iterator[Tuple[int, int]]:
    if d1 < d0:
        d0, d1 = d1, d0
    y, m = d0.year, d0.month
    y_end, m_end = d1.year, d1.month
    while (y, m) < (y_end, m_end) or (y == y_end and m == m_end):
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def iter_week_starts(d0: date, d1: date) -> Iterator[date]:
    """One frame per 7-day window starting at d0 until the window start passes d1."""
    if d1 < d0:
        d0, d1 = d1, d0
    cur = d0
    while cur <= d1:
        yield cur
        cur = cur + timedelta(days=7)


def timeseries_video(req: VideoRequest):
    init_ee()
    if not ee_runtime.EE_READY:
        raise HTTPException(
            status_code=500,
            detail=f"Earth Engine is not ready: {ee_runtime.EE_ERROR}",
        )

    da_chk = parse_iso_date(req.date_a or "")
    db_chk = parse_iso_date(req.date_b or "")
    if not (da_chk and db_chk) and req.year_a > req.year_b:
        raise HTTPException(status_code=400, detail="year_a must be <= year_b")

    cadence = (req.cadence or "monthly").strip().lower()
    if cadence not in ("monthly", "weekly"):
        cadence = "monthly"

    city_name, lat, lon = resolve_city(req.city)

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(req.radius_m).bounds()

    d_start, d_end = resolve_video_date_range(req)
    frames = []

    if cadence == "weekly":
        for week_start in iter_week_starts(d_start, d_end):
            week_end = week_start + timedelta(days=7)
            start_iso = week_start.isoformat()
            end_iso = week_end.isoformat()
            label = week_start.isoformat()
            try:
                frame = download_dw_frame(region, start_iso, end_iso, req.size, label)
                frames.append(frame)
            except Exception as e:
                print(f"Skipping frame week {label}: {e}")
    else:
        for y, m in iter_months_inclusive(d_start, d_end):
            try:
                frame = download_month_frame(region, y, m, req.size)
                frames.append(frame)
            except Exception as e:
                print(f"Skipping frame {y}-{m:02d}: {e}")

    if not frames:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate any {cadence} frames.",
        )

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

    filename = (
        f"timeseries_{cadence}_{safe_city}_{d_start.isoformat()}_{d_end.isoformat()}.mp4"
    )
    return FileResponse(
        tmp_path,
        media_type="video/mp4",
        filename=filename,
    )
