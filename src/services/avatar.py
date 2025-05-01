from datetime import datetime
import os
from fastapi import UploadFile

from src.models.avatars import Avatar, AvatarList
from src.db.database import Database


class AvatarService:
    def __init__(self, config: dict[str, str],  db: Database):
        self.config = config
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

        # Create avatar directory if it doesn't exist
        avatar_dir = os.path.join(self.config["avatars_path"], avatar_id)
        os.makedirs(avatar_dir, exist_ok=True)
        
        # Save photo file
        photo_path = os.path.join(avatar_dir, photo.filename)
        photo_content = await photo.read()
        with open(photo_path, "wb") as f:
            f.write(photo_content)
        
        # Save voice file
        voice_path = os.path.join(avatar_dir, voice.filename)
        voice_content = await voice.read()
        with open(voice_path, "wb") as f:
            f.write(voice_content)
        
        # Reset file positions for potential future reads
        await photo.seek(0)
        await voice.seek(0)
        
        avatar = Avatar(
            id=avatar_id,
            name=name,
            bio=bio,
            photo=os.path.join(avatar_id, photo.filename),  # Store relative path
            voice=os.path.join(avatar_id, voice.filename),  # Store relative path
            idle_video_url=f"https://example.com/videos/{avatar_id}/idle.mp4",
            created_at=datetime.now(),
        )

        return await self.db.create_avatar(avatar)

    def avatar_exists(self, avatar_id: str) -> bool:
        """Check if avatar exists."""
        return self.db.avatar_exists(avatar_id)
