from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Response

from src.containers.containers import AppContainer
from src.services.csm.csm import CSM
from src.services.ditto.ditto import Ditto

router = APIRouter()


@router.get("/")
def get_name() -> Response:
    return Response("Avatars")


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


@router.get("/health_check")
def health_check() -> Response:
    return Response("OK")
