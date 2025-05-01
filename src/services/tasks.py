from datetime import datetime
import uuid
import asyncio
from fastapi import HTTPException, BackgroundTasks

from src.models.tasks import Task
from src.services.csm.csm import CSM
from src.services.ditto.ditto import Ditto
from src.db.database import Database


class TaskService:
    def __init__(self, db: Database, tts: CSM, talking_head: Ditto):
        self.db = db
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
        await self.db.create_task(task)

        # Start background processing
        background_tasks.add_task(
            self._process_avatar_generation, task_id, avatar_id, text
        )

        return task

    async def get_task(self, task_id: str) -> Task:
        """Get task by ID."""
        task = await self.db.get_task(task_id)

        if task.status == "failed":
            raise HTTPException(status_code=500, detail=task.error)

        return task

    async def _process_avatar_generation(self, task_id: str, avatar_id: str, text: str):
        """Process avatar generation (runs in background)."""
        # Get task from database
        task = await self.db.get_task(task_id)
        try:
            # Update task to processing status
            task.status = "processing"
            await self.db.update_task(task)

            # Step 1: Generate audio using CSM (text-to-speech)
            await self.db.update_task(task)
            audio_path = self.tts.run(text)
            await asyncio.sleep(2)  # Simulate processing time

            # Step 2: Generate video using Ditto (talking head)
            await self.db.update_task(task)
            _ = self.talking_head.run(audio_path)
            await asyncio.sleep(3)  # Simulate processing time

            # Update task as completed
            task.status = "completed"
            task.video_url = (
                f"https://example.com/videos/{avatar_id}/generated/{task_id}.mp4"
            )
            await self.db.update_task(task)

        except Exception as e:
            # Handle failure
            task.status = "failed"
            task.error = str(e)
            await self.db.update_task(task)
