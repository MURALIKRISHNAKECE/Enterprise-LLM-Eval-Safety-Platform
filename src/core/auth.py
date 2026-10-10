"""
JWT Authentication + Database-backed users.
Provides login, token generation, and current-user dependency.
"""
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Header
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.config import settings
from src.core.database import get_db, User, APIKey

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_minutes: int | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.JWT_EXPIRE_MINUTES
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# ---------------------------------------------------------------------------
# API Key helpers
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key, key_hash, key_prefix)."""
    raw = "llmeval-" + secrets.token_urlsafe(32)
    key_hash   = hashlib.sha256(raw.encode()).hexdigest()
    key_prefix = raw[:12]
    return raw, key_hash, key_prefix


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# FastAPI dependency — get current user from JWT Bearer or API key
# ---------------------------------------------------------------------------

async def get_current_user(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str]    = Header(default=None),
    db: AsyncSession             = Depends(get_db),
) -> User:
    """
    Accepts either:
      - Authorization: Bearer <jwt>
      - X-API-Key: <raw_api_key>
    Falls back to anonymous viewer if neither provided.
    """
    # 1. Try JWT Bearer
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        payload = decode_token(token)
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Token missing subject.")
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive.")
        return user

    # 2. Try API Key
    if x_api_key:
        key_hash = hash_api_key(x_api_key)
        result = await db.execute(
            select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active == True)
        )
        api_key_obj = result.scalar_one_or_none()
        if not api_key_obj:
            raise HTTPException(status_code=401, detail="Invalid API key.")
        # Check expiry
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="API key expired.")
        # Load the user
        result2 = await db.execute(select(User).where(User.id == api_key_obj.user_id))
        user = result2.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User inactive.")
        return user

    # 3. Anonymous — viewer only
    return User(id=None, username="anonymous", role="viewer", is_active=True)


async def get_current_user_optional(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str]    = Header(default=None),
    db: AsyncSession             = Depends(get_db),
) -> Optional[User]:
    try:
        return await get_current_user(authorization, x_api_key, db)
    except HTTPException:
        return None


# ---------------------------------------------------------------------------
# Seed default admin user on startup (idempotent)
# ---------------------------------------------------------------------------

async def seed_default_admin(db: AsyncSession) -> None:
    """Creates admin/admin123 if no users exist. Remove in production."""
    result = await db.execute(select(User).limit(1))
    if result.scalar_one_or_none():
        return  # already seeded

    admin = User(
        username        = "admin",
        email           = "admin@llmeval.local",
        hashed_password = hash_password("admin123"),
        role            = "admin",
        is_active       = True,
    )
    db.add(admin)
    await db.commit()
    print("[AUTH] Default admin created: admin / admin123")