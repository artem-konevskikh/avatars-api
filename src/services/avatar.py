from datetime import datetime
from fastapi import UploadFile

from src.models.avatars import Avatar, AvatarList
from src.db.database import Database


class AvatarService:
    def __init__(self, db: Database):
        self.db = db

    async def get_avatars(self) -> AvatarList:
        """Get all avatars."""
        avatars = await self.db.get_all_avatars()
        return {"avatars": avatars}

    async def get_avatar(self, avatar_id: str) -> Avatar:
        """Get avatar by ID."""
        return await self.db.get_avatar(avatar_id)

    async def create_avatar(
        self, avatar_id: str, name: str, bio: str, photo: UploadFile, voice: UploadFile
    ) -> Avatar:
        """Create a new avatar."""
        avatar = Avatar(
            id=avatar_id,
            name=name,
            bio=bio,
            photo=photo.filename,  # In real implementation, this would be a path to stored file
            idle_video_url=f"https://example.com/videos/{avatar_id}/idle.mp4",
            created_at=datetime.now(),
        )

        return await self.db.create_avatar(avatar)

    def avatar_exists(self, avatar_id: str) -> bool:
        """Check if avatar exists."""
        return self.db.avatar_exists(avatar_id)
