"""Pydantic schemas for the TTS API."""

from typing import Optional

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Request body for POST /tts."""

    text: str = Field(..., min_length=1, max_length=5000)
    lang_code: str = Field(
        ..., description="Language code from languages.yaml (e.g. swa, amh, en)"
    )
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech rate multiplier")


class TTSResponse(BaseModel):
    """Response from POST /tts."""

    audio_base64: str = Field(..., description="Base64-encoded WAV audio")
    format: str = Field(default="wav")
    sample_rate: int = Field(default=22050)
    lang_code: str
    engine: str = Field(..., description="Engine used: espeak or mms")


class RatingRequest(BaseModel):
    """Request body for POST /ratings — create or update a rating."""

    reviewer: str = Field(
        ..., min_length=1, max_length=100, description="Reviewer name or ID"
    )
    language: str = Field(..., min_length=1, max_length=20, description="Language code")
    phrase: str = Field(
        ..., min_length=1, max_length=5000, description="The text that was synthesized"
    )
    rating: int = Field(
        ..., ge=1, le=5, description="Quality rating from 1 (worst) to 5 (best)"
    )
    comment: Optional[str] = Field(
        None, max_length=1000, description="Optional free-text comment"
    )
    audio_file: Optional[str] = Field(
        None, description="Path to reference audio file (if any)"
    )


class RatingResponse(BaseModel):
    """A single rating record returned from the API."""

    reviewer: str
    language: str
    phrase: str
    rating: int
    comment: Optional[str]
    timestamp: str
    audio_file: Optional[str]


class TokenResponse(BaseModel):
    """Response from POST /auth/login."""

    access_token: str
    token_type: str = "bearer"
    is_admin: bool = False


class UserRecord(BaseModel):
    """A user record returned from admin endpoints (no password hash)."""

    username: str
    is_admin: bool
    created_at: str


class CreateUserRequest(BaseModel):
    """Request body for POST /admin/users."""

    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)
    is_admin: bool = False


class ResetPasswordRequest(BaseModel):
    """Request body for POST /admin/users/{username}/reset-password."""

    password: str = Field(..., min_length=1)
    token_type: str = "bearer"
