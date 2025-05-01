from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, UploadFile, File, Form

from src.containers.containers import AppContainer
from src.services.avatar_service import AvatarService
from src.models.avatars import Avatar, AvatarList

router = APIRouter(prefix="/api", tags=["avatars"])


@router.get("/avatars", response_model=AvatarList)
@inject
async def list_avatars(
    avatar_service: AvatarService = Depends(Provide[AppContainer.avatar_service]),
):
    """List all avatars."""
    return await avatar_service.get_avatars()


@router.get("/avatar/{avatar_id}", response_model=Avatar)
@inject
async def get_avatar(
    avatar_id: str,
    avatar_service: AvatarService = Depends(Provide[AppContainer.avatar_service]),
):
    """Get avatar by ID."""
    return await avatar_service.get_avatar(avatar_id)


@router.post("/avatar/add", response_model=Avatar)
@inject
async def add_avatar(
    avatar_id: str = Form(...),
    name: str = Form(...),
    bio: str = Form(...),
    photo: UploadFile = File(...),
    voice: UploadFile = File(...),
    avatar_service: AvatarService = Depends(Provide[AppContainer.avatar_service]),
):
    """Create a new avatar."""
    return await avatar_service.create_avatar(avatar_id, name, bio, photo, voice)
