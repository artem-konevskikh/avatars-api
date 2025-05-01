from typing import Dict, List
from fastapi import HTTPException

from src.models.avatars import Avatar
from src.models.tasks import Task


class Database:
    """
    A dummy in-memory database service.
    In a real implementation, this would connect to PostgreSQL or another database.
    """

    def __init__(self):
        self.avatars: Dict[str, Avatar] = {}
        self.tasks: Dict[str, Task] = {}

    # Avatar operations
    async def get_all_avatars(self) -> List[Avatar]:
        """Get all avatars from the database."""
        return list(self.avatars.values())

    async def get_avatar(self, avatar_id: str) -> Avatar:
        """Get an avatar by ID."""
        if avatar_id not in self.avatars:
            raise HTTPException(status_code=404, detail="Avatar not found")
        return self.avatars[avatar_id]

    async def create_avatar(self, avatar: Avatar) -> Avatar:
        """Insert a new avatar into the database."""
        if avatar.id in self.avatars:
            raise HTTPException(status_code=400, detail="Avatar ID already exists")
        self.avatars[avatar.id] = avatar
        return avatar

    async def update_avatar(self, avatar: Avatar) -> Avatar:
        """Update an existing avatar."""
        if avatar.id not in self.avatars:
            raise HTTPException(status_code=404, detail="Avatar not found")
        self.avatars[avatar.id] = avatar
        return avatar

    async def delete_avatar(self, avatar_id: str) -> None:
        """Delete an avatar by ID."""
        if avatar_id not in self.avatars:
            raise HTTPException(status_code=404, detail="Avatar not found")
        del self.avatars[avatar_id]

    def avatar_exists(self, avatar_id: str) -> bool:
        """Check if an avatar exists."""
        return avatar_id in self.avatars

    # Task operations
    async def create_task(self, task: Task) -> Task:
        """Insert a new task into the database."""
        self.tasks[task.task_id] = task
        return task

    async def get_task(self, task_id: str) -> Task:
        """Get a task by ID."""
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return self.tasks[task_id]

    async def update_task(self, task: Task) -> Task:
        """Update an existing task."""
        if task.task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        self.tasks[task.task_id] = task
        return task

    async def delete_task(self, task_id: str) -> None:
        """Delete a task by ID."""
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        del self.tasks[task_id]
