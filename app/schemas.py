from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)
    timestamp: int | None = None
    content: str = Field(min_length=1)

    @field_validator("role", "content")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)


class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    created_at: str


class SearchResponse(BaseModel):
    data: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_ready: bool
