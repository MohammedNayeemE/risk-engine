from enum import Enum
from operator import add
from typing import Annotated, Any, List, Literal, Optional, Set, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# TODO: replace this with redis cache
IMAGE_HASH_SET: Set[str] = set()


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


class OverallState(TypedDict):
    user: UserProfile
    image_base64: str
    image_mime: str
    isValid: bool
    image_quality: bool
    image_hash: str
    image_quality_reason: str
    medicinelist: Annotated[Any, add]
    decision: Decision
    ban_hits: Annotated[List[str], add]
    nlem_hits: Annotated[List[str], add]
    nlem_misses: Annotated[List[str], add]
    explanation: MessagesState


# TODO: improve the input and outputstate
class InputState(TypedDict):
    user: UserProfile
    image_base64: str
    image_mime: str
    image_hash: str


class OutputState(TypedDict):
    explanation: MessagesState
