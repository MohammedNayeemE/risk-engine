from operator import add
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document


def merge_dicts(left: dict, right: dict) -> dict:
    """Deep merge dicts - right values override left, preserving nested structures"""
    if not left:
        return right
    if not right:
        return left
    
    merged = left.copy()
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Deep merge for nested dicts (e.g., medicine properties)
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


class IngestionState(TypedDict):
    input_path: str
    raw_documents: list[Document]
    normalised_text: list[Document]
    chunks: Annotated[list[dict], add]
    medicine_name: str
    extracted_fields: Annotated[dict[str, dict], merge_dicts]
    medicine_schemas: Annotated[dict[str, dict], merge_dicts]
    validation_warnings: Annotated[dict[str, list[str]], merge_dicts]
