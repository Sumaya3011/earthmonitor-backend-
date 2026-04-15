from fastapi import APIRouter, HTTPException

from schemas.requests import ChangeBody, ReportBody
from services.change_detection_service import compute_change_detection
from services.report_service import build_structured_report


router = APIRouter()


@router.post("/api/change-detection")
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


@router.post("/api/report")
def api_report(body: ReportBody):
    try:
        return build_structured_report(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
