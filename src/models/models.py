from pydantic import BaseModel


class VersionResponse(BaseModel):
    version: str
    build_date: str


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
