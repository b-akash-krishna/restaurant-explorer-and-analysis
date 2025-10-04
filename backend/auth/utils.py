import os
from datetime import datetime, timedelta
from typing import Optional, Annotated

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database import get_db # Changed from 'from ..database import get_db'
from backend.models import User as DBUser # Changed from 'from ..models import User as DBUser'

# --- JWT Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configure passlib with bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

def _truncate_password(password: str) -> str:
    """
    Truncate password to 72 bytes for bcrypt compatibility.
    Bcrypt has a maximum password length of 72 bytes.
    """
    if isinstance(password, str):
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate at byte level and decode back
            return password_bytes[:72].decode('utf-8', errors='ignore')
    return password

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    plain_password = _truncate_password(plain_password)
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    password = _truncate_password(password)
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(db: Session, username: str) -> Optional[DBUser]:
    """Retrieve a user from the database by username."""
    return db.query(DBUser).filter(DBUser.username == username).first()

def get_user_from_token(token: str, db: Session):
    """Extract user information from JWT token and find in database."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = get_user_by_username(db, username=username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency to get the current user
async def get_current_active_user(
    token: str,
    db: Session = Depends(get_db)
):
    """Dependency that gets the current active user."""
    user = get_user_from_token(token, db)
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return user

# Dependency to get the current active admin user
def get_current_active_admin_user(current_user: Annotated[DBUser, Depends(get_current_active_user)]):
    """Dependency that checks for admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource"
        )
    return current_user