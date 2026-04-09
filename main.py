# main.py
import json
import os
import io
import math
import tempfile
from datetime import date, timedelta
from typing import Optional

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
    DW_MIN_DATE,
)
from gee_utils import (
    WORLD_GEOM,
    regional_geom,
    get_dw_tile_urls_for_geometry,
    tile_url_for_day,
)
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
        service_account_json = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
        if not service_account_json:
            raise RuntimeError(
                "EE_SERVICE_ACCOUNT_JSON is missing. "
                "Set it in Render → Environment."
            )

        info = json.loads(service_account_json)
        email = info["client_email"]
        project_id = info.get("project_id")

        credentials = ee.ServiceAccountCredentials(email, key_data=service_account_json)
        if project_id:
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(credentials)

        EE_READY = True
        EE_ERROR = None
        print("Earth Engine initialized successfully.")
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

    if mode == "home":
        if not req.city or not str(req.city).strip():
            url = tile_url_for_day(WORLD_GEOM, da)
            return {
                "city": "World",
                "center_lat": 15.0,
                "center_lon": 0.0,
                "date_a": da.isoformat(),
                "date_b": da.isoformat(),
                "date_a_display": display_date(da),
                "date_b_display": display_date(da),
                "mode": mode,
                "tiles": {"a": url, "b": None, "change": None},
                "map_zoom": 2,
            }

        city_name, lat, lon = resolve_city(req.city)
        point = ee.Geometry.Point([lon, lat])
        geom = regional_geom(point)
        url = tile_url_for_day(geom, da)
        return {
            "city": city_name,
            "center_lat": lat,
            "center_lon": lon,
            "date_a": da.isoformat(),
            "date_b": da.isoformat(),
            "date_a_display": display_date(da),
            "date_b_display": display_date(da),
            "mode": mode,
            "tiles": {"a": url, "b": None, "change": None},
            "map_zoom": 11,
        }

    city_name, lat, lon = resolve_city(req.city)
    point = ee.Geometry.Point([lon, lat])
    geom = regional_geom(point)
    tiles = get_dw_tile_urls_for_geometry(geom, da, db)

    return {
        "city": city_name,
        "center_lat": lat,
        "center_lon": lon,
        "date_a": da.isoformat(),
        "date_b": db.isoformat(),
        "date_a_display": display_date(da),
        "date_b_display": display_date(db),
        "mode": mode,
        "tiles": tiles,
    }


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
            "The app uses calendar dates (day–month–year) for Dynamic World daily mosaics. "
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
