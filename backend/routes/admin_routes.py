from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from backend.auth import require_role
from backend.database.mongodb import get_database
from backend.security.audit_logger import log_security_event

router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(require_role(["admin"]))],
)


class RoleUpdateRequest(BaseModel):
    role: str


@router.get("/stats")
async def get_system_stats() -> dict[str, Any]:
    """Returns high-level system, user, and security statistics."""
    database = get_database()
    now = datetime.now(timezone.utc)

    total_users = database.users.count_documents({})
    admin_users = database.users.count_documents({"role": "admin"})
    patient_users = database.users.count_documents({"role": "patient"})

    locked_users = database.users.count_documents({
        "locked_until": {"$gt": now},
    })

    total_audit_events = database.audit_logs.count_documents({})
    failed_logins = database.audit_logs.count_documents({"action": "LOGIN_FAILED_PASSWORD"})
    lockout_events = database.audit_logs.count_documents({"action": "ACCOUNT_LOCKED"})

    return {
        "total_users": total_users,
        "admin_users": admin_users,
        "patient_users": patient_users,
        "locked_users": locked_users,
        "total_audit_events": total_audit_events,
        "failed_logins": failed_logins,
        "lockout_events": lockout_events,
    }


@router.get("/users")
async def list_users(limit: int = Query(default=50, ge=1, le=100)) -> list[dict[str, Any]]:
    """
    Lists users for administrative management.
    Zero-knowledge enforcement: never exposes patient-encrypted PII (names, Aadhaar, PAN).
    """
    database = get_database()
    now = datetime.now(timezone.utc)

    cursor = database.users.find(
        {},
        {
            "_id": 1,
            "role": 1,
            "is_active": 1,
            "failed_login_attempts": 1,
            "locked_until": 1,
            "last_login_at": 1,
            "created_at": 1,
        },
    ).sort("created_at", -1).limit(limit)

    users_list = []
    for doc in cursor:
        locked_until = doc.get("locked_until")
        is_locked = False
        remaining_seconds = 0

        if locked_until and isinstance(locked_until, datetime):
            if locked_until.tzinfo is None:
                locked_until = locked_until.replace(tzinfo=timezone.utc)
            if locked_until > now:
                is_locked = True
                remaining_seconds = int((locked_until - now).total_seconds())

        users_list.append({
            "id": str(doc.get("_id")),
            "role": doc.get("role", "patient"),
            "is_active": doc.get("is_active", True),
            "failed_login_attempts": doc.get("failed_login_attempts", 0),
            "is_locked": is_locked,
            "remaining_lockout_seconds": remaining_seconds,
            "last_login_at": doc.get("last_login_at").isoformat() if isinstance(doc.get("last_login_at"), datetime) else None,
            "created_at": doc.get("created_at").isoformat() if isinstance(doc.get("created_at"), datetime) else None,
        })

    return users_list


@router.post("/users/{target_user_id}/unlock")
async def unlock_user_account(
    target_user_id: str,
    request: Request,
    admin_id: ObjectId = Depends(require_role(["admin"])),
) -> dict[str, str]:
    """Manually clears account lockout and resets failed login counters."""
    client_ip = request.client.host if request.client else None
    try:
        user_obj_id = ObjectId(target_user_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid target user ID") from exc

    database = get_database()
    result = database.users.update_one(
        {"_id": user_obj_id},
        {
            "$set": {
                "failed_login_attempts": 0,
                "locked_until": None,
                "active_session_id": None,
                "session_expires_at": None,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Target user not found")

    log_security_event(
        action="ADMIN_ACCOUNT_UNLOCKED",
        user_id=admin_id,
        details={"target_user_id": target_user_id},
        ip_address=client_ip,
        status="SUCCESS",
    )

    return {"message": f"User account {target_user_id} successfully unlocked"}


@router.post("/users/{target_user_id}/role")
async def update_user_role(
    target_user_id: str,
    payload: RoleUpdateRequest,
    request: Request,
    admin_id: ObjectId = Depends(require_role(["admin"])),
) -> dict[str, str]:
    """Updates a user's role (admin or patient)."""
    client_ip = request.client.host if request.client else None
    new_role = payload.role.strip().lower()

    if new_role not in {"admin", "patient"}:
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'patient'")

    try:
        user_obj_id = ObjectId(target_user_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid target user ID") from exc

    database = get_database()
    result = database.users.update_one(
        {"_id": user_obj_id},
        {
            "$set": {
                "role": new_role,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Target user not found")

    log_security_event(
        action="ADMIN_ROLE_UPDATED",
        user_id=admin_id,
        details={"target_user_id": target_user_id, "new_role": new_role},
        ip_address=client_ip,
        status="SUCCESS",
    )

    return {"message": f"User {target_user_id} role updated to '{new_role}'"}


@router.get("/audit-logs")
async def get_global_audit_logs(limit: int = Query(default=100, ge=1, le=100)) -> list[dict[str, Any]]:
    """Retrieves system-wide immutable audit trail logs."""
    database = get_database()
    cursor = database.audit_logs.find(
        {},
        {"_id": 1, "action": 1, "user_id": 1, "status": 1, "details": 1, "ip_address": 1, "timestamp": 1},
    ).sort("timestamp", -1).limit(limit)

    logs = []
    for doc in cursor:
        ts = doc.get("timestamp")
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts) if ts else ""

        logs.append({
            "id": str(doc.get("_id")),
            "action": doc.get("action", ""),
            "user_id": str(doc.get("user_id", "ANONYMOUS")),
            "status": doc.get("status", "SUCCESS"),
            "details": doc.get("details", {}),
            "ip_address": doc.get("ip_address", "UNKNOWN"),
            "timestamp": ts_str,
        })

    return logs
