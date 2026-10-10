"""
RBAC - Role-Based Access Control
Roles: Admin, Evaluator, Reviewer, Viewer
"""
from fastapi import Header, HTTPException
from enum import Enum


class Role(str, Enum):
    ADMIN     = "admin"
    EVALUATOR = "evaluator"
    REVIEWER  = "reviewer"
    VIEWER    = "viewer"


ROLE_MAP: dict[str, Role] = {
    "admin-key-001":     Role.ADMIN,
    "evaluator-key-001": Role.EVALUATOR,
    "reviewer-key-001":  Role.REVIEWER,
    "viewer-key-001":    Role.VIEWER,
}

PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {
        "run_evaluation",
        "view_history",
        "submit_feedback",
        "run_golden",
        "view_drift",
        "view_latency",
        "compare_models",
        "delete_logs",
        "manage_users",
    },
    Role.EVALUATOR: {
        "run_evaluation",
        "view_history",
        "run_golden",
        "view_drift",
        "view_latency",
        "compare_models",
    },
    Role.REVIEWER: {
        "view_history",
        "submit_feedback",
        "view_drift",
        "view_latency",
    },
    Role.VIEWER: {
        "view_history",
        "view_drift",
        "view_latency",
    },
}


def get_role(x_api_key: str = Header(default=None)) -> Role:
    """
    FastAPI dependency - extracts role from X-API-Key header.
    Returns Role.VIEWER if no key provided (read-only access).
    Raises 401 if key is invalid.
    """
    if x_api_key is None:
        return Role.VIEWER
    role = ROLE_MAP.get(x_api_key)
    if role is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Check X-API-Key header.",
        )
    return role


def check_permission(role: Role, permission: str) -> bool:
    """Check if a role has a specific permission. Returns bool."""
    return permission in PERMISSIONS.get(role, set())