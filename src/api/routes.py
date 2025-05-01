from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Response

from src.containers.containers import AppContainer
from src.services.csm.csm import CSM
from src.services.ditto.ditto import Ditto
from src.models.models import VersionResponse, HealthResponse

router = APIRouter()


@router.get("/", response_model=VersionResponse)
async def get_version():
    return {"version": "0.1.0", "build_date": "2025-05-01"}


@router.get("/run")
@inject
def run(
    text: str,
    tts: CSM = Depends(Provide[AppContainer.tts]),
    talking_head: Ditto = Depends(Provide[AppContainer.talking_head]),
) -> Response:
    res = ""
    res += tts.run(text)
    res += "\n"
    res += talking_head.run(text)
    return Response(res)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "database": "connected",
            "task_queue": "operational",
            "csm_service": "available",
            "ditto_service": "available"
        }
    }