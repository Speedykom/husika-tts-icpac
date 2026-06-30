"""FastAPI TTS service for HUSIKA — Sprint 1 skeleton."""

import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt

from ..auth import UserStore
from ..engines.engine_router import EngineRouter
from ..ratings import RatingsStore
from .schemas import (
    CreateUserRequest,
    RatingRequest,
    RatingResponse,
    ResetPasswordRequest,
    TokenResponse,
    TTSRequest,
    TTSResponse,
    UserRecord,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auth configuration
_JWT_SECRET = os.environ.get("JWT_SECRET", "change-before-deploying")
_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 30

_API_KEY = os.environ.get("API_KEY", "")

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
_user_store = UserStore()


def _create_token(username: str) -> str:
    expire = datetime.now(tz=timezone.utc) + timedelta(
        minutes=_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(
        {"sub": username, "exp": expire}, _JWT_SECRET, algorithm=_JWT_ALGORITHM
    )


def _validate_jwt(token: str) -> Optional[dict]:
    """Return user dict if token is valid, else None."""
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if not username:
            return None
        return _user_store.get_user(username)
    except JWTError:
        return None


async def _require_auth(token: str = Depends(_oauth2_scheme)) -> dict:
    """JWT-only auth — used for the login endpoint itself."""
    user = _validate_jwt(token) if token else None
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def _require_any_auth(request: Request) -> None:
    """Accept either a valid X-API-Key header or a valid Bearer JWT."""
    api_key = request.headers.get("X-API-Key")
    if api_key and _API_KEY and api_key == _API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        if _validate_jwt(auth[7:]):
            return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Provide a valid X-API-Key header or Bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


_DESCRIPTION = """\
Multilingual Text-to-Speech for the IGAD borderlands early warning system.

## Authentication

Pass your API key in every request:

```
X-API-Key: hsk_your_key_here
```

Generate a key once with:

```bash
python scripts/generate_api_key.py
```

Then add `API_KEY=hsk_...` to your `.env` file.

Alternatively, obtain a short-lived JWT via `POST /auth/login` and send it as\
 `Authorization: Bearer <token>`.

## Supported Languages

| Code | Language    | Engine |
|------|-------------|--------|
| `swa` | Swahili    | MMS    |
| `amh` | Amharic    | MMS    |
| `ara` | Arabic     | MMS    |
| `som` | Somali     | MMS    |
| `orm` | Oromo      | MMS    |
| `tir` | Tigrinya   | MMS    |
| `lug` | Luganda    | MMS    |
| `kin` | Kinyarwanda | MMS   |
| `rn`  | Kirundi    | MMS    |
| `nue` | Nuer       | MMS    |
| `din` | Dinka      | MMS    |
| `pko` | Pokoot     | MMS    |
| `tuv` | Turkana    | Coqui VITS |
| `kdj` | Karamojong | Coqui VITS |
| `en`  | English    | eSpeak |
| `fr`  | French     | eSpeak |

## Audio Format

Responses contain a base64-encoded **16-bit PCM WAV** file. Decode with:

```python
import base64, wave, io
audio = base64.b64decode(response["audio_base64"])
# audio is a valid WAV binary — write to file or stream directly
```
"""

app = FastAPI(
    title="Husika TTS API",
    description=_DESCRIPTION,
    version="0.1.0",
    openapi_tags=[
        {"name": "TTS", "description": "Text-to-speech synthesis endpoints"},
        {"name": "Auth", "description": "Obtain a short-lived JWT Bearer token"},
        {
            "name": "Ratings",
            "description": "Submit and query synthesis quality ratings",
        },
        {"name": "Health", "description": "Service health and language metadata"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict origins for production
    allow_methods=["*"],
    allow_headers=["*"],
)

router = EngineRouter()
ratings_store = RatingsStore()


def _custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": (
                "Paste **only the key value** "
                "(e.g. `hsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`). "
                "Do **not** include `API_KEY=`. "
                "Generate a new key with: `python scripts/generate_api_key.py`."
            ),
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Short-lived JWT obtained from `POST /auth/login`.",
        },
    }
    schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi

# Auth endpoints


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Exchange username + password for a Bearer JWT."""
    user = _user_store.authenticate(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(
        access_token=_create_token(user["username"]),
        is_admin=bool(user.get("is_admin", False)),
    )


async def _require_admin(user: dict = Depends(_require_auth)) -> dict:
    """FastAPI dependency — validates JWT and asserts the user is an admin."""
    if not user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


# Serve the web-ui directory at /ui
web_ui_path = Path(__file__).resolve().parent.parent.parent / "web-ui"
if web_ui_path.is_dir():
    app.mount("/ui", StaticFiles(directory=str(web_ui_path), html=True), name="web-ui")


@app.post(
    "/tts",
    response_model=TTSResponse,
    tags=["TTS"],
    summary="Synthesize speech",
    response_description="Base64-encoded WAV audio with synthesis metadata",
)
async def synthesize(
    request: TTSRequest,
    _auth: None = Depends(_require_any_auth),
) -> TTSResponse:
    """Synthesize speech from text in the given language."""
    try:
        engine, lang_code = router.select_engine(request.lang_code)
        result = await engine.synthesize(
            text=request.text,
            language_code=lang_code,
            speed=request.speed,
        )
        return TTSResponse(
            audio_base64=result["audio_base64"],
            format=result["format"],
            sample_rate=result["sample_rate"],
            lang_code=request.lang_code,
            engine=result["engine"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Synthesis failed")


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    """Returns `{"status": "ok"}` when the service is running."""
    return {"status": "ok", "service": "husika-tts"}


@app.get("/languages", tags=["Health"], summary="List supported languages")
async def list_languages() -> dict:
    """Returns metadata for every supported language including engine and model info."""
    return router.get_language_info()


@app.post(
    "/ratings",
    response_model=RatingResponse,
    status_code=200,
    tags=["Ratings"],
    summary="Submit a quality rating",
)
async def submit_rating(
    request: RatingRequest,
    _auth: None = Depends(_require_any_auth),
) -> RatingResponse:
    """Record a quality rating for a synthesized phrase (history is kept)."""
    record = ratings_store.add(
        reviewer=request.reviewer,
        language=request.language,
        phrase=request.phrase,
        rating=request.rating,
        comment=request.comment,
        audio_file=getattr(request, "audio_file", None),
    )
    return RatingResponse(
        reviewer=record["reviewer"],
        language=record["language"],
        phrase=record["phrase"],
        rating=int(record["rating"]),
        comment=record["comment"] or None,
        timestamp=record["timestamp"],
        audio_file=record.get("audio_file"),
    )


@app.get(
    "/ratings",
    response_model=list[RatingResponse],
    tags=["Ratings"],
    summary="Query ratings",
)
async def get_ratings(
    language: Optional[str] = Query(None, description="Filter by language code"),
    reviewer: Optional[str] = Query(None, description="Filter by reviewer name"),
    phrase: Optional[str] = Query(None, description="Filter by exact phrase"),
    _auth: None = Depends(_require_any_auth),
) -> list[RatingResponse]:
    """Query stored ratings, optionally filtered."""
    rows = ratings_store.query(language=language, reviewer=reviewer, phrase=phrase)
    return [
        RatingResponse(
            reviewer=r["reviewer"],
            language=r["language"],
            phrase=r["phrase"],
            rating=int(r["rating"]),
            comment=r["comment"] or None,
            timestamp=r["timestamp"],
            audio_file=r.get("audio_file"),
        )
        for r in rows
    ]


# ── Admin endpoints ──────────────────────────────────────────────────────────


@app.get("/admin/users", response_model=list[UserRecord])
async def admin_list_users(_admin: dict = Depends(_require_admin)) -> list[UserRecord]:
    """List all users (no password hashes)."""
    return [UserRecord(**u) for u in _user_store.list_users()]


@app.post("/admin/users", response_model=UserRecord, status_code=201)
async def admin_create_user(
    request: CreateUserRequest, _admin: dict = Depends(_require_admin)
) -> UserRecord:
    """Create a new user."""
    try:
        result = _user_store.create_user(
            username=request.username,
            password=request.password,
            is_admin=request.is_admin,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    user = _user_store.get_user(result["username"])
    return UserRecord(
        username=user["username"],
        is_admin=user["is_admin"],
        created_at=user["created_at"],
    )


@app.post("/admin/users/{username}/reset-password", status_code=204)
async def admin_reset_password(
    username: str,
    request: ResetPasswordRequest,
    _admin: dict = Depends(_require_admin),
) -> None:
    """Set a new password for the given user."""
    if not _user_store.update_password(username, request.password):
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")


@app.delete("/admin/users/{username}", status_code=204)
async def admin_delete_user(
    username: str, admin: dict = Depends(_require_admin)
) -> None:
    """Delete a user. An admin cannot delete their own account."""
    if username.lower() == admin["username"].lower():
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    if not _user_store.delete_user(username):
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")


@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...), _user: dict = Depends(_require_auth)
):
    """Accept an audio file upload and store it in /data/uploads."""
    uploads_dir = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    # Use a timestamped filename to avoid collisions
    ext = Path(file.filename).suffix or ".wav"
    fname = f"audio_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S%f')}{ext}"
    dest = uploads_dir / fname
    with dest.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    return {"audio_file": f"uploads/{fname}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
