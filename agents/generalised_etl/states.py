from operator import add
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts — values from right override left"""
    if not left:
        return right
    merged = left.copy()
    merged.update(right)
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
