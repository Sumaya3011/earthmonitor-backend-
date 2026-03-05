import json
import os

import ee
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from pydantic import BaseModel

from config import YEARS, LOCATION_LAT, LOCATION_LON, LOCATION_NAME
from gee_utils import get_dw_tile_urls   # your existing logic
from chat_utils import ask_chatbot       # OpenAI helper


# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Earth Engine init
# ---------------------------
def init_ee():
    """
    Initialize Earth Engine using a service account JSON stored in
    environment variable EE_SERVICE_ACCOUNT_JSON.
    """
    if getattr(init_ee, "_done", False):
        return

    service_account_json = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise RuntimeError(
            "EE_SERVICE_ACCOUNT_JSON env var is missing. "
            "On Render, set it in the Environment tab."
        )

    info = json.loads(service_account_json)
    email = info["client_email"]
    project_id = info.get("project_id")

    credentials = ee.ServiceAccountCredentials(email, key_data=service_account_json)
    if project_id:
        ee.Initialize(credentials, project=project_id)
    else:
        ee.Initialize(credentials)

    init_ee._done = True


init_ee()


# ---------------------------
# Pydantic models
# ---------------------------
class MapRequest(BaseModel):
    mode: str          # "change_detection" | "single_year" | "timeseries"
    year_a: int
    year_b: int
    city: str | None = None


class ChatRequest(BaseModel):
    message: str
    mode: str
    year_a: int
    year_b: int
    city: str | None = None


# ---------------------------
# Helpers
# ---------------------------
geolocator = Nominatim(user_agent="dw-change-app")


def resolve_city(city: str | None):
    """Return (name, lat, lon). Fallback to defaults from config.py."""
    if not city or not city.strip():
        return LOCATION_NAME, LOCATION_LAT, LOCATION_LON

    try:
        loc = geolocator.geocode(city.strip())
        if loc:
            return city.strip(), loc.latitude, loc.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass

    # Fallback to default location if geocoding fails
    return LOCATION_NAME, LOCATION_LAT, LOCATION_LON


# ---------------------------
# API endpoints
# ---------------------------
@app.post("/map-config")
def map_config(req: MapRequest):
    """
    Returns map center + Dynamic World tile URLs for the selected mode/years/city.
    Frontend calls this whenever the user changes settings.
    """
    mode = req.mode
    year_a = req.year_a
    year_b = req.year_b

    # Clamp years to valid values
    if year_a not in YEARS:
        year_a = YEARS[0]
    if year_b not in YEARS:
        year_b = YEARS[-1]

    city_name, lat, lon = resolve_city(req.city)
    location_point = ee.Geometry.Point([lon, lat])

    # Use your existing gee_utils.get_dw_tile_urls logic
    if mode == "single_year":
        tiles = get_dw_tile_urls(location_point, year_a, year_a)
    else:
        tiles = get_dw_tile_urls(location_point, year_a, year_b)

    return {
        "city": city_name,
        "center_lat": lat,
        "center_lon": lon,
        "year_a": year_a,
        "year_b": year_b,
        "mode": mode,
        "tiles": tiles,   # {"a": "...", "b": "...", "change": "..."}
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Chat endpoint: the frontend sends the question + map settings.
    We ask OpenAI and return the reply + a summary for the bottom bar.
    """
    city_name, lat, lon = resolve_city(req.city)

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that explains Dynamic World land cover "
            "maps and changes over time using SIMPLE language. "
            "The app has three analysis modes: change_detection, single_year, timeseries. "
            f"Current mode: {req.mode}, years: {req.year_a}–{req.year_b}. "
            f"Current region: {city_name} at ({lat:.3f}, {lon:.3f}). "
            "Explain clearly what the map likely shows and any key patterns "
            "for the selected area and time range."
        ),
    }

    messages_for_api = [
        system_msg,
        {"role": "user", "content": req.message},
    ]

    reply = ask_chatbot(messages_for_api)

    return {
        "reply": reply,
        "summary": reply,   # same text for the bottom summary
    }
