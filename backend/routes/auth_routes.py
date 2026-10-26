from __future__ import annotations

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.security.rate_limiter import limiter

from backend.auth import get_authenticated_user_id
from backend.database.mongodb import get_database
from backend.security.audit_logger import get_user_security_activity, log_security_event
from backend.services.auth_service import login_user, logout_user, register_user
from backend.services.token_service import (
    TokenRevocationUnavailable,
    blacklist_token,
    create_token_pair,
    verify_refresh_token,
)
from backend.settings import get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])

REFRESH_COOKIE_NAME = "refresh_token"
REFRESH_COOKIE_PATH = "/api/auth"
REFRESH_COOKIE_MAX_AGE = 7 * 24 * 60 * 60


def _set_refresh_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        max_age=REFRESH_COOKIE_MAX_AGE,
        httponly=True,
        secure=getattr(settings, "refresh_cookie_secure", False),
        samesite="lax",
        path=REFRESH_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=getattr(settings, "refresh_cookie_secure", False),
        samesite="lax",
    )


def _refresh_error(detail: object, status_code: int = 401) -> JSONResponse:
    # Raising HTTPException after modifying an injected Response can discard
    # Set-Cookie headers. Return a response directly so the browser receives
    # the cookie deletion on invalid/expired refresh sessions.
    response = JSONResponse(
        status_code=status_code,
        content={"detail": detail},
    )
    _clear_refresh_cookie(response)
    return response


class RegisterRequest(BaseModel):
    phone: str
    password: str
    consent_given: bool = False


class LoginRequest(BaseModel):
    phone: str
    password: str


class PageVisitRequest(BaseModel):
    page: str


@router.post("/register", status_code=201)
@limiter.limit("3/minute")
async def register(request: Request, payload: RegisterRequest) -> dict[str, str]:
    client_ip = request.client.host if request.client else None
    try:
        user_id = register_user(
            payload.phone,
            payload.password,
            payload.consent_given,
            ip_address=client_ip,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user_id": str(user_id), "message": "User registered successfully"}


@router.post("/login")
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    payload: LoginRequest,
) -> dict[str, object]:
    client_ip = request.client.host if request.client else None
    try:
        result = login_user(payload.phone, payload.password, ip_address=client_ip)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    # Keep the existing access-token response shape, but never expose the
    # refresh token to JavaScript. The browser stores it as an HttpOnly cookie.
    if isinstance(result, str):
        return {"access_token": result, "token_type": "bearer"}

    refresh_token = result.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=500, detail="Authentication did not issue a refresh token")

    _set_refresh_cookie(response, refresh_token)
    return {key: value for key, value in result.items() if key != "refresh_token"}


@router.post("/refresh")
@limiter.limit("10/minute")
async def refresh_tokens(request: Request, response: Response) -> dict[str, object]:
    # CSRF defense for cookie-authenticated refresh: browser Origin must match
    # the origin serving this API. Non-browser clients may omit Origin.
    origin = request.headers.get("origin")
    expected_origin = str(request.base_url).rstrip("/")
    if origin and origin.rstrip("/") != expected_origin:
        return _refresh_error("Cross-origin refresh request rejected", status_code=403)
    refresh_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not refresh_token:
        return _refresh_error("Refresh session is missing. Please sign in again.")

    try:
        token_data = verify_refresh_token(refresh_token)
    except TokenRevocationUnavailable:
        raise HTTPException(status_code=503, detail="Authentication service temporarily unavailable")
    except ValueError:
        return _refresh_error("Invalid or expired refresh session. Please sign in again.")

    user_id = token_data.get("sub")
    role = token_data.get("role", "patient")
    sid = token_data.get("sid")
    if not user_id or not sid:
        return _refresh_error("Invalid refresh-token payload")

    try:
        user_object_id = ObjectId(str(user_id))
    except Exception:
        return _refresh_error("Invalid user identity in refresh token")

    try:
        database = get_database()
        user_doc = database.users.find_one(
            {"_id": user_object_id},
            {"active_session_id": 1, "is_active": 1, "role": 1},
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Authentication service temporarily unavailable") from exc
    # A refresh token is valid only while its session is still the active session.
    # In particular, active_session_id=None after logout must not revive the session.
    if (
        not user_doc
        or user_doc.get("is_active") is False
        or user_doc.get("active_session_id") != sid
    ):
        return _refresh_error({
            "error": "SESSION_SUPERSEDED",
            "message": "Your session is no longer active. Please sign in again.",
        })

    # Use the current database role, not a potentially stale role claim.
    role = user_doc.get("role", "patient")

    # Do not issue a replacement token if revocation of the old one failed.
    if not blacklist_token(refresh_token):
        raise HTTPException(
            status_code=503,
            detail="Unable to rotate the refresh session. Please retry.",
        )

    new_tokens = create_token_pair(user_id, role=role, session_id=sid)
    _set_refresh_cookie(response, new_tokens["refresh_token"])
    return {key: value for key, value in new_tokens.items() if key != "refresh_token"}


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    user_id: ObjectId = Depends(get_authenticated_user_id),
) -> dict[str, str]:
    client_ip = request.client.host if request.client else None
    auth_token = getattr(request.state, "auth_token", None)
    refresh_token = request.cookies.get(REFRESH_COOKIE_NAME)

    refresh_revoked = True
    if refresh_token:
        refresh_revoked = blacklist_token(refresh_token)

    logout_user(auth_token, user_id=user_id, ip_address=client_ip)
    _clear_refresh_cookie(response)

    if not refresh_revoked:
        raise HTTPException(
            status_code=503,
            detail="Refresh session revocation failed. Please retry logout.",
        )

    return {"message": "Logged out successfully and session revoked"}


@router.get("/security/activity")
async def security_activity(
    limit: int = Query(default=100, ge=1, le=100),
    user_id: ObjectId = Depends(get_authenticated_user_id),
) -> dict[str, object]:
    return get_user_security_activity(user_id, limit=limit)


@router.post("/security/log-visit")
async def log_page_visit(
    request: Request,
    payload: PageVisitRequest,
    user_id: ObjectId = Depends(get_authenticated_user_id),
) -> dict[str, str]:
    client_ip = request.client.host if request.client else None
    page_key = payload.page.strip().lower()
    action_map = {
        "dashboard": "DASHBOARD_VIEW",
        "documents": "DOCUMENTS_PAGE_VIEW",
        "upload": "DOCUMENTS_PAGE_VIEW",
        "autofill": "FORMS_PAGE_VIEW",
        "forms": "FORMS_PAGE_VIEW",
        "profile": "PROFILE_VIEW",
        "activity": "SECURITY_ACTIVITY_VIEW",
        "admin": "ADMIN_CONSOLE_VIEW",
    }
    action = action_map.get(page_key, f"{page_key.upper()}_PAGE_VIEW")
    log_security_event(
        action=action,
        user_id=user_id,
        details={"page": page_key},
        ip_address=client_ip,
        status="SUCCESS",
    )
    return {"status": "ok", "action": action}
