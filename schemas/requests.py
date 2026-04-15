from typing import Optional

from pydantic import BaseModel


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
    # "monthly" | "weekly" — frames from date_a through date_b when both ISO dates are sent;
    # otherwise year_a/year_b define the range (Jan 1 year_a .. Dec 31 year_b).
    cadence: str = "monthly"
    date_a: Optional[str] = None
    date_b: Optional[str] = None


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
