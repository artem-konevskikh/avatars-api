from fastapi import APIRouter

from src.models.models import VersionResponse, HealthResponse

router = APIRouter()


@router.get("/", response_model=VersionResponse)
async def get_version():
    return {"version": "0.1.0", "build_date": "2025-05-01"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "database": "connected",
            "task_queue": "operational",
            "csm_service": "available",
            "ditto_service": "available",
        },
    }
