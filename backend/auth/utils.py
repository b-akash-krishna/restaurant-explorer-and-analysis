from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
import os

# --- JWT Configuration ---
# NOTE: The SECRET_KEY should be loaded from an environment variable.
# Example: SECRET_KEY = os.environ.get("SECRET_KEY")
SECRET_KEY = "your-super-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configure passlib with bcrypt
# Set truncate_error to False to allow automatic truncation of passwords
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


def get_user_from_token(token: str):
    """Extract user information from JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        # Check if the 'role' is present in the payload
        role: str = payload.get("role")

        if username is None or role is None:
            return None
        return username, role
    except JWTError:
        return None