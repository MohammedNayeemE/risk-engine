"""Utility functions for API key generation and management."""
import hashlib
import secrets
from typing import Tuple


def generate_api_key() -> str:
    """
    Generate a secure random API key.
    
    Format: re_<32 random hex characters>
    Example: re_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
    
    Returns:
        A securely generated API key string
    """
    random_part = secrets.token_hex(32)
    return f"re_{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using SHA-256.
    
    Args:
        api_key: The plain text API key to hash
        
    Returns:
        The hashed API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def get_key_prefix(api_key: str) -> str:
    """
    Extract the prefix (first 8 characters) from an API key for display/identification.
    
    Args:
        api_key: The API key
        
    Returns:
        First 8 characters of the key
    """
    return api_key[:8] if len(api_key) >= 8 else api_key


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verify that a provided API key matches the stored hash.
    
    Args:
        provided_key: The API key provided by the client
        stored_hash: The hashed API key from the database
        
    Returns:
        True if the key matches, False otherwise
    """
    return hash_api_key(provided_key) == stored_hash


def generate_api_key_with_hash() -> Tuple[str, str, str]:
    """
    Generate a new API key and return the key, its hash, and prefix.
    
    Returns:
        Tuple of (api_key, key_hash, key_prefix)
    """
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    key_prefix = get_key_prefix(api_key)
    return api_key, key_hash, key_prefix
