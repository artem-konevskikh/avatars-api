from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Task(BaseModel):
    task_id: str
    avatar_id: str
    text: str
    status: str
    created_at: datetime
    video_url: Optional[str] = None
    error: Optional[str] = None


class TaskCreate(BaseModel):
    avatar_id: str
    text: str
