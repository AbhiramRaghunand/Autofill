from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from bson import ObjectId
from pymongo import ASCENDING

from backend.database.mongodb import get_database
from backend.settings import get_settings

logger = logging.getLogger(__name__)


class TokenRevocationUnavailable(RuntimeError):
    """Raised when revocation status cannot be determined safely."""


ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


def setup_token_blacklist_ttl_index() -> None:
    """Ensures MongoDB TTL index exists on token_blacklist.expires_at to auto-expire entries."""
    try:
        database = get_database()
        database.token_blacklist.create_index(
            [("expires_at", ASCENDING)],
            expireAfterSeconds=0,
            background=True,
        )
    except Exception as exc:
        logger.warning("Could not ensure TTL index on token_blacklist: %s", exc)


def create_token_pair(
    user_id: str | ObjectId,
    role: str = "patient",
    session_id: str | None = None,
) -> dict[str, str]:
    """Generates a short-lived access token and a long-lived refresh token with role and session claims."""
    settings = get_settings()
    secret = settings.jwt_secret_key or "development-secret-change-me"
    algorithm = settings.jwt_algorithm or "HS256"
    now = datetime.now(timezone.utc)

    sid = session_id or str(uuid.uuid4())

    access_jti = str(uuid.uuid4())
    access_exp = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_payload = {
        "sub": str(user_id),
        "role": str(role or "patient"),
        "sid": sid,
        "jti": access_jti,
        "type": "access",
        "iat": now,
        "exp": access_exp,
    }
    access_token = jwt.encode(access_payload, secret, algorithm=algorithm)

    refresh_jti = str(uuid.uuid4())
    refresh_exp = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_payload = {
        "sub": str(user_id),
        "role": str(role or "patient"),
        "sid": sid,
        "jti": refresh_jti,
        "type": "refresh",
        "iat": now,
        "exp": refresh_exp,
    }
    refresh_token = jwt.encode(refresh_payload, secret, algorithm=algorithm)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "role": str(role or "patient"),
        "session_id": sid,
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


def is_jti_blacklisted(jti: str) -> bool:
    """Checks if a token's jti is in the revocation blacklist."""
    if not jti:
        return False
    try:
        database = get_database()
        return database.token_blacklist.find_one({"jti": jti}) is not None
    except Exception as exc:
        logger.error("Failed to query token blacklist; rejecting authentication", exc_info=True)
        raise TokenRevocationUnavailable("Token revocation status is unavailable") from exc


def blacklist_token(token: str) -> bool:
    """Extracts jti and exp from token and marks it revoked in MongoDB."""
    if not token:
        return False
    settings = get_settings()
    secret = settings.jwt_secret_key or "development-secret-change-me"
    algorithm = settings.jwt_algorithm or "HS256"

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        if not jti:
            return False

        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            expires_at = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
        else:
            expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        database = get_database()
        database.token_blacklist.update_one(
            {"jti": jti},
            {
                "$set": {
                    "jti": jti,
                    "user_id": payload.get("sub"),
                    "token_type": payload.get("type", "unknown"),
                    "revoked_at": datetime.now(timezone.utc),
                    "expires_at": expires_at,
                }
            },
            upsert=True,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to blacklist token: %s", exc)
        return False


def verify_refresh_token(refresh_token: str) -> dict[str, Any]:
    """Validates signature, expiration, and blacklist status of a refresh token."""
    settings = get_settings()
    secret = settings.jwt_secret_key or "development-secret-change-me"
    algorithm = settings.jwt_algorithm or "HS256"

    try:
        payload = jwt.decode(refresh_token, secret, algorithms=[algorithm])
    except jwt.InvalidTokenError as exc:
        raise ValueError("Invalid or expired refresh token") from exc

    if payload.get("type") != "refresh":
        raise ValueError("Provided token is not a refresh token")

    jti = payload.get("jti")
    if not jti:
        raise ValueError("Refresh token is missing its identifier")
    if is_jti_blacklisted(jti):
        raise ValueError("Refresh token has been revoked")

    return payload
