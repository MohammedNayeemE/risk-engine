from operator import add
from typing import Annotated, Any, List, Optional, TypedDict

from pydantic import BaseModel


class UserProfile(TypedDict):
    name: str
    age: str
    allergies: Optional[List[str]]
    gender: str
    known_conditions: Optional[List[str]]


class OverallState(TypedDict):
    user: UserProfile
    image_base64: str
    image_mime: str
    isValid: bool
    image_quality: bool
    image_quality_reason: str
    medicinelist: Annotated[Any, add]


class InputState(TypedDict):
    user: UserProfile
    image_base64: str
    image_mime: str


class OutputState(TypedDict):
    isValid: bool
