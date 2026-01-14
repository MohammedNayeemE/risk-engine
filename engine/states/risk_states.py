from enum import Enum
from operator import add
from typing import Annotated, Any, List, Literal, Optional, Set, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class MedicineItem(BaseModel):
    name: Optional[str] = Field(
        default=None, description="Name of the medicine (e.g., Paracetamol)"
    )

    form: Optional[Literal["tablet", "capsule", "liquid", "syringe"]] = Field(
        default=None, description="Dosage form of the medicine"
    )

    strength: Optional[str] = Field(
        default=None,
        description="Amount of active drug per unit (e.g., 50 mg per tablet)",
    )

    dose: Optional[str] = Field(
        default=None, description="Amount taken at one time (e.g., 1 tablet, 10 ml)"
    )

    dosage: Optional[str] = Field(
        default=None,
        description="Complete dosing schedule (e.g., 1 tablet every 8 hours for 5 days)",
    )


class MedicineList(BaseModel):
    """List of extracted medicines"""

    medicinelist: List[MedicineItem]


class UserProfile(TypedDict):
    name: str
    age: str
    allergies: Optional[List[str]]
    gender: str
    known_conditions: Optional[List[str]]


class Decision(Enum):
    FAIL = "fail"
    WARN = "warn"
    PASS = "pass"


class MedicineSafetyIssue(BaseModel):
    """Represents a safety concern found during evaluation"""
    medicine_name: str
    issue_type: Literal["age_contraindication", "dosage_unsafe", "allergy_contraindication", "drug_interaction", "special_condition_contraindication", "other"]
    severity: Literal["critical", "warning", "info"]
    description: str
    recommendation: Optional[str] = None


class SafetyEvaluation(BaseModel):
    """Result of safety evaluation for a medicine"""
    medicine_name: str
    decision: Decision
    issues: List[MedicineSafetyIssue] = []
    age_appropriate: bool = True
    dosage_safe: bool = True
    allergy_safe: bool = True
    interaction_safe: bool = True
    condition_compatible: bool = True


class AdminReport(BaseModel):
    """Comprehensive report for admin/pharmacist review"""
    overall_decision: Decision
    patient_age: int
    patient_gender: str
    medicines_evaluated: List[str]
    safety_evaluations: List[SafetyEvaluation] = []
    banned_findings: List[str] = []
    critical_issues: List[MedicineSafetyIssue] = []
    warning_issues: List[MedicineSafetyIssue] = []
    requires_approval: bool = False
    report_reason: str
    recommendations: List[str] = []


class OverallState(TypedDict):
    user: UserProfile
    image_base64: str
    image_mime: str
    isValid: bool
    image_quality: bool
    image_hash: str
    image_quality_reason: str
    medicinelist: Annotated[Any, add]
    missing_medicines: list[str]
    missing_fields: dict[str, list[str]]
    web_search_results: dict[str, Any]
    db_medicine_data: dict[str, dict]
    safety_evaluations: Annotated[List[SafetyEvaluation], add]
    decision: Decision
    ban_hits: Annotated[List[str], add]
    safety_issues: Annotated[List[MedicineSafetyIssue], add]
    admin_report: Optional[AdminReport]
    explanation: MessagesState


# TODO: improve the input and outputstate
class InputState(TypedDict):
    user: UserProfile
    image_base64: str
    image_mime: str
    image_hash: str


class OutputState(TypedDict):
    explanation: MessagesState
