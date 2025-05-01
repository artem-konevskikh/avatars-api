from datetime import datetime
from typing import Dict
from fastapi import HTTPException, UploadFile

from src.models.avatars import Avatar, AvatarList


class AvatarService:
    def __init__(self):
        # This would be a database in a real implementation
        self.avatars_db: Dict[str, Avatar] = {}

    async def get_avatars(self) -> AvatarList:
        """Get all avatars."""
        return self.avatars_db.values()

    async def get_avatar(self, avatar_id: str) -> Avatar:
        """Get avatar by ID."""
        if avatar_id not in self.avatars_db:
            raise HTTPException(status_code=404, detail="Avatar not found")

        return self.avatars_db[avatar_id]

    async def create_avatar(
        self, avatar_id: str, name: str, bio: str, photo: UploadFile, voice: UploadFile
    ) -> dict:
        """Create a new avatar."""
        if avatar_id in self.avatars_db:
            raise HTTPException(status_code=400, detail="Avatar ID already exists")

        avatar = Avatar(
            id=avatar_id,
            name=name,
            bio=bio,
            photo=photo.filename,
            idle_video_url=f"https://example.com/videos/{avatar_id}/idle.mp4",
            created_at=datetime.now(),
        )
        self.avatars_db[avatar_id] = avatar

        return avatar

    def avatar_exists(self, avatar_id: str) -> bool:
        """Check if avatar exists."""
        return avatar_id in self.avatars_db
