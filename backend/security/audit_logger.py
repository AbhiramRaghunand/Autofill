from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from backend.database.mongodb import get_database

logger = logging.getLogger(__name__)


def log_security_event(
    action: str,
    user_id: ObjectId | str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
    status: str = "SUCCESS",
) -> None:
    """Appends an immutable security audit event to the MongoDB audit_logs collection."""
    try:
        database = get_database()
        user_val = str(user_id) if user_id else "ANONYMOUS"
        event = {
            "action": action,
            "user_id": user_val,
            "status": status,
            "details": details or {},
            "ip_address": ip_address or "UNKNOWN",
            "timestamp": datetime.now(timezone.utc),
        }
        database.audit_logs.insert_one(event)
    except Exception as exc:
        logger.warning("Failed to record security audit log: %s", exc)


def get_user_security_activity(user_id: ObjectId | str, limit: int = 100) -> dict[str, Any]:
    """Retrieves total count and recent security activity for a user."""
    limit = max(1, min(int(limit), 100))
    try:
        database = get_database()
        user_str = str(user_id)
        total_count = database.audit_logs.count_documents({"user_id": user_str})
        cursor = database.audit_logs.find(
            {"user_id": user_str},
            {"_id": 0, "action": 1, "status": 1, "details": 1, "ip_address": 1, "timestamp": 1},
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
                "action": doc.get("action", ""),
                "status": doc.get("status", "SUCCESS"),
                "details": doc.get("details", {}),
                "ip_address": doc.get("ip_address", "UNKNOWN"),
                "timestamp": ts_str,
            })
        return {"total": total_count, "logs": logs}
    except Exception as exc:
        logger.warning("Failed to read user audit logs: %s", exc)
        return {"total": 0, "logs": []}
