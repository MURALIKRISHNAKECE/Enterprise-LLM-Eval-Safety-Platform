"""
Auth routes — login, register, API key management.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.database import get_db, User, APIKey
from src.core.auth import (
    hash_password, verify_password, create_access_token,
    generate_api_key, get_current_user,
)
from src.core.rbac import check_permission, Role

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "viewer"


class APIKeyCreateRequest(BaseModel):
    label: str = "default"
    role: str  = "evaluator"
    expires_days: int | None = None


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------
@router.post("/login")
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == payload.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled.")

    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    token = create_access_token({"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "role":         user.role,
        "username":     user.username,
    }


# ---------------------------------------------------------------------------
# POST /register  (admin only)
# ---------------------------------------------------------------------------
@router.post("/register")
async def register(
    payload: RegisterRequest,
    db: AsyncSession      = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    if not check_permission(Role(current_user.role), "manage_users"):
        raise HTTPException(status_code=403, detail="Only admins can register users.")

    valid_roles = {"admin", "evaluator", "reviewer", "viewer"}
    if payload.role not in valid_roles:
        raise HTTPException(status_code=422, detail=f"Role must be one of: {valid_roles}")

    # Check duplicate
    result = await db.execute(select(User).where(User.username == payload.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already exists.")

    user = User(
        username        = payload.username,
        email           = payload.email,
        hashed_password = hash_password(payload.password),
        role            = payload.role,
        is_active       = True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return {"status": "created", "user_id": user.id, "username": user.username, "role": user.role}


# ---------------------------------------------------------------------------
# POST /api-keys  — generate API key for current user
# ---------------------------------------------------------------------------
@router.post("/api-keys")
async def create_api_key(
    payload: APIKeyCreateRequest,
    db: AsyncSession   = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.id is None:
        raise HTTPException(status_code=401, detail="Must be logged in to create API keys.")

    from datetime import timedelta
    expires_at = None
    if payload.expires_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=payload.expires_days)

    raw_key, key_hash, key_prefix = generate_api_key()

    api_key = APIKey(
        user_id    = current_user.id,
        key_hash   = key_hash,
        key_prefix = key_prefix,
        role       = payload.role,
        is_active  = True,
        expires_at = expires_at,
    )
    db.add(api_key)
    await db.commit()

    return {
        "api_key":    raw_key,   # shown once only
        "key_prefix": key_prefix,
        "role":       payload.role,
        "expires_at": expires_at,
        "warning":    "Store this key securely — it will not be shown again.",
    }


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------
@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "user_id":    current_user.id,
        "username":   current_user.username,
        "role":       current_user.role,
        "is_active":  current_user.is_active,
        "last_login": current_user.last_login,
    }


# ---------------------------------------------------------------------------
# GET /users  (admin only)
# ---------------------------------------------------------------------------
@router.get("/users")
async def list_users(
    db: AsyncSession   = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not check_permission(Role(current_user.role), "manage_users"):
        raise HTTPException(status_code=403, detail="Admins only.")

    result = await db.execute(select(User))
    users = result.scalars().all()
    return [
        {
            "id":         u.id,
            "username":   u.username,
            "email":      u.email,
            "role":       u.role,
            "is_active":  u.is_active,
            "created_at": u.created_at,
            "last_login": u.last_login,
        }
        for u in users
    ]