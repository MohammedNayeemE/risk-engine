from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field

from engine.states.risk_states import MedicineItem


class UserProfileRequest(BaseModel):
    """Patient profile payload."""

    name: str = Field(..., description="Patient name")
    age: str = Field(..., description="Patient age")
    gender: str = Field(..., description="Patient gender (M/F)")
    allergies: Optional[list[str]] = Field(default=[], description="List of allergies")
    known_conditions: Optional[list[str]] = Field(
        default=[], description="List of known conditions"
    )


class RiskAssessmentRequest(BaseModel):
    """Request body for risk assessment."""

    user_profile: UserProfileRequest
    image_base64: str = Field(..., description="Base64 encoded image of prescription")
    image_mime: str = Field(default="image/jpeg", description="MIME type of the image")


class RiskAssessmentResponse(BaseModel):
    """Response after graph completes."""

    status: str
    user_name: str
    medicines_found: int
    medicines: list[MedicineItem] = []
    image_hash: str
    is_valid: bool
    image_quality: bool
    image_quality_reason: str
    web_search_results: dict = {}
    missing_medicines: list[str] = []
    missing_fields: dict = {}
    raw_state: dict = {}


class HumanApprovalResponse(BaseModel):
    """Response when awaiting human approval."""

    status: str = "awaiting_approval"
    thread_id: str
    user_name: str
    medicines: list[MedicineItem] = []
    message: str
    approval_data: dict = {}


class ApprovalRequest(BaseModel):
    """Payload to approve, regenerate, or edit a paused assessment."""

    thread_id: str
    action: str = Field(
        ..., description="Action to take: 'approve', 'regenerate', or 'edit'"
    )
    edited_medicines: Optional[list[MedicineItem]] = Field(
        None, description="Edited medicine list if action is 'edit'"
    )


class HealthResponse(BaseModel):
    """Simple health payload."""

    status: str = "healthy"
    service: str = "Risk Engine API"


# ---------------------------------------------------------------------------
# Client Registration & API Key Management Schemas
# ---------------------------------------------------------------------------


class ClientRegistrationRequest(BaseModel):
    """Request payload for registering a new client."""

    company_name: str = Field(..., min_length=1, description="Company name")
    email: EmailStr = Field(..., description="Client email address")
    domain: str = Field(
        ..., min_length=1, description="Frontend domain (e.g., https://example.com)"
    )


class ClientRegistrationResponse(BaseModel):
    """Response after successful client registration."""

    client_id: int
    company_name: str
    email: str
    domain: str
    api_key: str
    message: str = "Client registered successfully. Please save your API key securely."


class CreateAPIKeyRequest(BaseModel):
    """Request payload for creating a new API key."""

    name: str = Field(default="API Key", description="Optional name for the API key")


class CreateAPIKeyResponse(BaseModel):
    """Response after creating a new API key."""

    api_key: str
    key_prefix: str
    name: str
    created_at: str
    message: str = "API key created successfully. Please save it securely."


class APIKeyInfoResponse(BaseModel):
    """Response showing API key information (without the actual key)."""

    id: int
    key_prefix: str
    name: Optional[str]
    is_active: bool
    last_used_at: Optional[str]
    created_at: str
    expires_at: Optional[str]


class ListAPIKeysResponse(BaseModel):
    """Response listing all API keys for a client."""

    api_keys: list[APIKeyInfoResponse]


class RevokeAPIKeyRequest(BaseModel):
    """Request payload for revoking an API key."""

    api_key_id: int = Field(..., description="ID of the API key to revoke")


class RevokeAPIKeyResponse(BaseModel):
    """Response after revoking an API key."""

    success: bool
    message: str
