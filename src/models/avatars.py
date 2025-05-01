from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class Avatar(BaseModel):
    id: str
    name: str
    bio: str
    photo: str
    idle_video_url: Optional[str] = None
    created_at: datetime


class AvatarList(BaseModel):
    avatars: List[Avatar]
