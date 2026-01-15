"""Database models for client management and API keys."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class Client(BaseModel):
    """Client model for registered API users."""
    
    id: Optional[int] = None
    company_name: str
    email: EmailStr
    domain: str
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class APIKey(BaseModel):
    """API Key model for authentication."""
    
    id: Optional[int] = None
    client_id: int
    key_hash: str
    key_prefix: str  # Store first 8 chars for identification
    name: Optional[str] = None  # Optional name for the key
    is_active: bool = True
    last_used_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class APIKeyWithSecret(APIKey):
    """API Key model with plain text secret (only returned on creation)."""
    
    api_key: str  # Full plain text key
