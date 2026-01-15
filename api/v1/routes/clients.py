"""API routes for client registration and API key management."""
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.v1.schemas import (
    APIKeyInfoResponse,
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    ListAPIKeysResponse,
    RevokeAPIKeyRequest,
    RevokeAPIKeyResponse,
)
from app.middleware.auth import get_current_client
from persistence.models import Client
from persistence.repositories import (
    create_api_key_for_client,
    create_client,
    get_client_by_email,
    list_client_api_keys,
    revoke_api_key,
)

router = APIRouter(prefix="/clients", tags=["Client Management"])


@router.post("/register", response_model=ClientRegistrationResponse)
async def register_client(request: ClientRegistrationRequest):
    """
    Register a new client and generate their first API key.

    This endpoint creates a new client account with:
    - Company name
    - Email address (must be unique)
    - Frontend domain for CORS validation

    Returns the client information along with an API key.
    **Important**: Save the API key securely as it won't be shown again.
    """
    # Check if client already exists
    existing_client = get_client_by_email(request.email)
    if existing_client:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Client with email {request.email} already exists",
        )

    try:
        # Create client and generate API key
        client, api_key_obj = create_client(
            company_name=request.company_name,
            email=request.email,
            domain=request.domain,
        )

        return ClientRegistrationResponse(
            client_id=client.id,
            company_name=client.company_name,
            email=client.email,
            domain=client.domain,
            api_key=api_key_obj.api_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register client: {str(e)}",
        )


@router.post("/api-keys", response_model=CreateAPIKeyResponse)
async def create_new_api_key(
    request: CreateAPIKeyRequest,
    current_client: Annotated[Client, Depends(get_current_client)],
):
    """
    Create a new API key for the authenticated client.

    Requires authentication with an existing API key.
    Useful for:
    - Creating keys with specific names
    - Rotating keys
    - Creating environment-specific keys (dev, staging, prod)

    **Important**: Save the API key securely as it won't be shown again.
    """
    try:
        api_key_obj = create_api_key_for_client(
            client_id=current_client.id, name=request.name
        )

        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )

        return CreateAPIKeyResponse(
            api_key=api_key_obj.api_key,
            key_prefix=api_key_obj.key_prefix,
            name=api_key_obj.name or "API Key",
            created_at=api_key_obj.created_at.isoformat() if api_key_obj.created_at else "",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}",
        )


@router.get("/api-keys", response_model=ListAPIKeysResponse)
async def list_api_keys(
    current_client: Annotated[Client, Depends(get_current_client)],
):
    """
    List all API keys for the authenticated client.

    Returns information about all API keys including:
    - Key prefix (for identification)
    - Name
    - Active status
    - Last used timestamp
    - Creation date

    Note: The full API key is never returned for security reasons.
    """
    try:
        api_keys = list_client_api_keys(current_client.id)

        return ListAPIKeysResponse(
            api_keys=[
                APIKeyInfoResponse(
                    id=key.id,
                    key_prefix=key.key_prefix,
                    name=key.name,
                    is_active=key.is_active,
                    last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
                    created_at=key.created_at.isoformat() if key.created_at else "",
                    expires_at=key.expires_at.isoformat() if key.expires_at else None,
                )
                for key in api_keys
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}",
        )


@router.delete("/api-keys", response_model=RevokeAPIKeyResponse)
async def revoke_key(
    request: RevokeAPIKeyRequest,
    current_client: Annotated[Client, Depends(get_current_client)],
):
    """
    Revoke (deactivate) an API key.

    The key will be marked as inactive and can no longer be used for authentication.
    This action cannot be undone - if you need access again, create a new API key.

    **Warning**: Make sure you have at least one active API key before revoking others,
    or you'll lose access to your account.
    """
    try:
        # Verify the key belongs to the current client
        api_keys = list_client_api_keys(current_client.id)
        key_to_revoke = next(
            (k for k in api_keys if k.id == request.api_key_id), None
        )

        if not key_to_revoke:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or does not belong to your account",
            )

        if not key_to_revoke.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="API key is already revoked",
            )

        # Check if this is the last active key
        active_keys = [k for k in api_keys if k.is_active and k.id != request.api_key_id]
        if len(active_keys) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot revoke the last active API key. Create a new key first.",
            )

        success = revoke_api_key(request.api_key_id)

        if success:
            return RevokeAPIKeyResponse(
                success=True,
                message="API key revoked successfully",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to revoke API key",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke API key: {str(e)}",
        )
