from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "uploads" / "temp"
RESULTS_DIR = ROOT_DIR / "uploads" / "documents"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_version: str = "v1alpha"
    max_upload_size_mb: int = 25
    upload_dir: Path = UPLOAD_DIR
    results_dir: Path = RESULTS_DIR
    mongodb_uri: str = ""
    mongodb_database: str = ""
    local_aes_secret_key: str = ""
    hmac_salt_secret: str = ""
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    refresh_cookie_secure: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        gemini_api_version=os.getenv("GEMINI_API_VERSION", "v1alpha").strip(),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")),
        mongodb_uri=os.getenv("MONGODB_URI", "").strip(),
        mongodb_database=os.getenv("MONGODB_DATABASE", "").strip(),
        local_aes_secret_key=os.getenv("LOCAL_AES_SECRET_KEY", "").strip(),
        hmac_salt_secret=os.getenv("HMAC_SALT_SECRET", "").strip(),
        jwt_secret_key=os.getenv("JWT_SECRET_KEY", "development-secret-change-me").strip(),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256").strip(),
        # Set REFRESH_COOKIE_SECURE=true in HTTPS deployments.
        refresh_cookie_secure=os.getenv("REFRESH_COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"},
    )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    return settings
