from sqlalchemy import Column, Integer, String, Boolean
from .database import Base

class User(Base):
    """SQLAlchemy model for the 'users' table."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True, nullable=True) # Optional field
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user") # 'user' or 'admin'