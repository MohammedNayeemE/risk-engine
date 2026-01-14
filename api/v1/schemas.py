from typing import Optional

from pydantic import BaseModel, Field


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


class MedicineItem(BaseModel):
    """Medicine item returned by extraction/graph."""

    name: Optional[str] = None
    form: Optional[str] = None
    strength: Optional[str] = None
    dose: Optional[str] = None
    dosage: Optional[str] = None


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
