from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from src.containers.containers import AppContainer
from src.services.avatar import AvatarService
from src.services.tasks import TaskService
from src.models.tasks import Task, TaskCreate

router = APIRouter(prefix="/api", tags=["tasks"])


@router.post("/avatar/generate", response_model=Task)
@inject
async def generate_avatar(
    task_data: TaskCreate,
    background_tasks: BackgroundTasks,
    avatar_service: AvatarService = Depends(Provide[AppContainer.avatar_service]),
    task_service: TaskService = Depends(Provide[AppContainer.task_service]),
) -> Task:
    """Create a task to generate avatar video from text."""
    avatar_id = task_data.avatar_id

    # Check if avatar exists
    if not avatar_service.avatar_exists(avatar_id):
        raise HTTPException(status_code=404, detail="Avatar not found")

    # Create task
    return await task_service.create_task(avatar_id, task_data.text, background_tasks)


@router.get("/avatar/tasks/{task_id}", response_model=Task)
@inject
async def get_task_status(
    task_id: str,
    task_service: TaskService = Depends(Provide[AppContainer.task_service]),
) -> Task:
    """Get task status by ID."""
    return await task_service.get_task(task_id)
