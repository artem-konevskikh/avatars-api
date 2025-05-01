from datetime import datetime
from typing import Dict
import uuid
import asyncio
from fastapi import HTTPException, BackgroundTasks


from src.models.tasks import Task
from src.services.csm.csm import CSM
from src.services.ditto.ditto import Ditto


class TaskService:
    def __init__(self, tts: CSM, talking_head: Ditto):
        # This would be a database in a real implementation
        self.tasks_db: Dict[str, Task] = {}
        self.tts = tts
        self.talking_head = talking_head

    async def create_task(
        self, avatar_id: str, text: str, background_tasks: BackgroundTasks
    ) -> Task:
        """Create a new task for avatar generation."""
        # Create a new task
        task_id = str(uuid.uuid4())
        created_at = datetime.now()

        task = Task(
            task_id=task_id,
            avatar_id=avatar_id,
            text=text,
            status="queued",
            created_at=created_at,
            progress=0.0,
        )

        # Store task in database
        self.tasks_db[task_id] = task

        # Start background processing
        background_tasks.add_task(
            self._process_avatar_generation, task_id, avatar_id, text
        )

        return {"task_id": task_id, "status": "queued", "created_at": created_at}

    async def get_task(self, task_id: str) -> Task:
        """Get task by ID."""
        if task_id not in self.tasks_db:
            raise HTTPException(status_code=404, detail="Task not found")

        task = self.tasks_db[task_id]

        if task.status == "failed":
            raise HTTPException(status_code=500, detail=task.error)

        return task

    async def _process_avatar_generation(self, task_id: str, avatar_id: str, text: str):
        """Process avatar generation (runs in background)."""
        try:
            # Update task to processing
            self.tasks_db[task_id].status = "processing"

            audio_path = self.tts.run(text)
            await asyncio.sleep(2)
            _ = self.talking_head.run(audio_path)
            await asyncio.sleep(3)

            self.tasks_db[task_id].status = "completed"
            self.tasks_db[task_id].video_url = (
                f"https://example.com/videos/{avatar_id}/generated/{task_id}.mp4",
            )

        except Exception as e:
            # Handle failure
            self.tasks_db[task_id].status = "failed"
            self.tasks_db[task_id].error = str(e)
