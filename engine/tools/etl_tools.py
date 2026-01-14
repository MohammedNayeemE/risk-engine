import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import psycopg
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers.pdf import _merge_text_and_extras
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Base64Bytes, BaseModel, Field

from engine.llm.models import ModelManager
from engine.prompts.etl_prompts import (
    age_restriction,
    agentic_chunking,
    dosage,
    identity,
)
from engine.prompts.etl_prompts import regulatory as regulatory_prompt
from engine.prompts.etl_prompts import strength

load_dotenv()


class FileLoader:
    """
    Factory class for loading types of documents.
    use load function to call the FileLoader class

    example:

    docs = FileLoader.load("src/something.pdf")
    """

    @staticmethod
    def load(filepath: Union[str, Path]):
        path = Path(filepath)
        if not path.exists():
            return None

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return FileLoader._loadpdf(path)
        else:
            return None

    @staticmethod
    def _loadpdf(path: Path) -> list[Document] | None:
        loader = PyPDFLoader(path)
        docs = loader.load()
        normalize_docs = []
        for doc in docs[19:125]:
            result = Document(
                page_content=doc.page_content.strip(), metadata=doc.metadata
            )
            normalize_docs.append(result)

        return normalize_docs


def _fix_line_breaks(text: str) -> str:
    """
    Joins lines that were broken due to PDF formatting,
    but preserves paragraph boundaries.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    fixed = []
    buffer = ""
    for line in lines:
        if buffer and re.search(r"[.;:)]$", buffer):
            fixed.append(buffer)
            buffer = line
        else:
            if buffer:
                buffer += " " + line
            else:
                buffer = line
    if buffer:
        fixed.append(buffer)
    return "\n\n".join(fixed)


def _remove_headers_footers(text: str) -> str:
    """
    Removes common noise like page numbers, copyrights.
    VERY conservative.
    """
    patterns = [
        r"^page\s+\d+$",
        r"©.*",
        r"all rights reserved",
    ]
    cleaned_lines = []
    for line in text.splitlines():
        l = line.strip().lower()
        if any(re.match(p, l) for p in patterns):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _final_cleanup(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_docs(documents: list[Document]) -> list[Document]:
    normalize_docs: list[Document] = []
    for doc in documents:
        text = doc.page_content
        text = _remove_headers_footers(text)
        text = _fix_line_breaks(text)
        text = _final_cleanup(text)

        normalize_docs.append(Document(page_content=text, metadata=doc.metadata))

    return normalize_docs


class Chunk(BaseModel):
    text: str
    medicine_name: Optional[str] = Field(
        description="Name of the medicine this chunk refers to, or None if unclear"
    )
    chunk_type: Literal[
        "identity",
        "strength",
        "dosage",
        "age_restriction",
        "regulatory",
        "other",
    ]
    confidence: float = Field(ge=0.0, le=1.0)


class ChunkList(BaseModel):
    chunks: List[Chunk]


class Dosage(BaseModel):
    raw_dosage_text: Optional[str]
    max_single_dose_mg: Optional[int]
    max_daily_dose_mg: Optional[int]
    min_interval_hours: Optional[int]
    confidence: float = Field(ge=0.0, le=1.0)


class Identity(BaseModel):
    name: str
    generic_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class Strength(BaseModel):
    strength: str  # e.g., "500 mg", "200 mg/5 ml", "10 mg per tablet"
    confidence: float = Field(ge=0.0, le=1.0)


class AgeRestriction(BaseModel):
    min_age: Optional[int]
    max_age: Optional[int]
    confidence: float = Field(ge=0.0, le=1.0)


class Regulatory(BaseModel):
    banned: Optional[bool]
    prescription_required: Optional[bool]
    confidence: float = Field(ge=0.0, le=1.0)


def _extract_with_llm(human_content: str, system_content: str, structure) -> dict:
    ollama_model = ModelManager.get_model_for_local_processing()
    response = ollama_model.with_structured_output(structure).invoke(
        [SystemMessage(content=system_content), HumanMessage(content=human_content)]
    )

    return response.model_dump()


def llm_chunk(documents: list[Document]) -> list[dict]:
    all_chunks: list[dict] = []
    for doc in documents:
        try:
            if len(all_chunks) >= 50:
                return all_chunks
            chunk_wrapper = _extract_with_llm(
                human_content=doc.page_content,
                system_content=agentic_chunking,
                structure=ChunkList,
            )
            chunks = chunk_wrapper["chunks"]
            all_chunks.extend(chunks)
        except Exception as e:
            continue

    return all_chunks


def extract_dosage(chunks: list[dict]) -> Any:
    extractions: list[dict] = []
    medicine_name = (chunks[0].get("medicine_name") if chunks else None) or "unknown"

    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        try:
            resp = _extract_with_llm(
                human_content=text,
                system_content=dosage,
                structure=Dosage,
            )
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    continue
            if isinstance(resp, dict):
                extractions.append(resp)
        except Exception as e:
            print(f"Dosage extraction error: {e}")
            continue

    if not extractions:
        return {
            "medicine_name": medicine_name,
            "raw_dosage_text": None,
            "max_single_dose_mg": None,
            "max_daily_dose_mg": None,
            "min_interval_hours": None,
            "confidence": 0.0,
        }

    best = max(extractions, key=lambda d: float(d.get("confidence", 0.0)))

    max_single = None
    max_daily = None
    min_interval = None

    for ext in extractions:
        try:
            if ext.get("max_single_dose_mg") is not None:
                max_single = (
                    max_single
                    if max_single is not None
                    else ext.get("max_single_dose_mg")
                )
                if ext.get("max_single_dose_mg") is not None:
                    max_single = max(max_single, ext.get("max_single_dose_mg"))

            if ext.get("max_daily_dose_mg") is not None:
                max_daily = (
                    max_daily if max_daily is not None else ext.get("max_daily_dose_mg")
                )
                if ext.get("max_daily_dose_mg") is not None:
                    max_daily = max(max_daily, ext.get("max_daily_dose_mg"))

            if ext.get("min_interval_hours") is not None:
                min_interval = (
                    min_interval
                    if min_interval is not None
                    else ext.get("min_interval_hours")
                )
                if ext.get("min_interval_hours") is not None:
                    min_interval = min(min_interval, ext.get("min_interval_hours"))
        except Exception:
            continue

    overall_conf = sum(float(e.get("confidence", 0.0)) for e in extractions) / len(
        extractions
    )

    return {
        "medicine_name": medicine_name,
        "raw_dosage_text": best.get("raw_dosage_text") or None,
        "max_single_dose_mg": max_single,
        "max_daily_dose_mg": max_daily,
        "min_interval_hours": min_interval,
        "confidence": round(overall_conf, 3),
    }


def extract_strength(chunks: list[dict], medicine_name: str = "unknown") -> dict:
    extractions: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        try:
            response = _extract_with_llm(
                human_content=text,
                system_content=strength,
                structure=Strength,
            )
            if isinstance(response, dict):
                extractions.append(response)
        except Exception as e:
            print(f"Strength extraction error: {e}")
            continue

    if not extractions:
        return {
            "medicine_name": medicine_name,
            "strength": "unknown",
            "pack_size": None,
            "confidence": 0.0,
        }

    best_strength = "unknown"
    best_pack = None
    best_strength_conf = -1.0

    for ext in extractions:
        strength_val = str(ext.get("strength", "")).strip()
        pack_val = ext.get("pack_size")
        conf = float(ext.get("confidence", 0.0))

        if strength_val and conf > best_strength_conf:
            best_strength = strength_val
            best_strength_conf = conf

        if pack_val:
            pack_val = str(pack_val).strip()
            if best_pack is None or conf > best_strength_conf:
                best_pack = pack_val

    overall_conf = sum(float(ext.get("confidence", 0.0)) for ext in extractions) / len(
        extractions
    )

    return {
        "medicine_name": medicine_name,
        "strength": best_strength,
        "pack_size": best_pack,
        "confidence": round(overall_conf, 3),
    }


def extract_age(chunks: list[dict]) -> Any:
    extractions: list[dict] = []
    medicine_name = (chunks[0].get("medicine_name") if chunks else None) or "unknown"

    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        try:
            resp = _extract_with_llm(
                human_content=text,
                system_content=age_restriction,
                structure=AgeRestriction,
            )
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    continue
            if isinstance(resp, dict):
                extractions.append(resp)
        except Exception as e:
            print(f"Age extraction error: {e}")
            continue

    if not extractions:
        return {
            "medicine_name": medicine_name,
            "min_age": None,
            "max_age": None,
            "confidence": 0.0,
        }

    min_age_values = [
        e.get("min_age") for e in extractions if e.get("min_age") is not None
    ]
    max_age_values = [
        e.get("max_age") for e in extractions if e.get("max_age") is not None
    ]

    min_age = max(min_age_values) if min_age_values else None
    max_age = min(max_age_values) if max_age_values else None

    overall_conf = sum(float(e.get("confidence", 0.0)) for e in extractions) / len(
        extractions
    )

    return {
        "medicine_name": medicine_name,
        "min_age": min_age,
        "max_age": max_age,
        "confidence": round(overall_conf, 3),
    }


def extract_identity(chunks: list[dict], medicine_name: Any = "unknown") -> Any:
    extractions: list[dict] = []
    for chunk in chunks:
        response = _extract_with_llm(
            human_content=chunk.get("text", ""),
            system_content=identity,
            structure=Identity,
        )
        if isinstance(response, dict):
            extractions.append(response)
    if not extractions:
        return {
            "medicine_name": medicine_name,
            "name": "unknown",
            "generic_name": "unknown",
            "confidence": 0.0,
        }
    best_name = "unknown"
    best_generic = "unknown"
    best_name_conf = -1.0
    best_generic_conf = -1.0

    for ext in extractions:
        name = str(ext.get("name", "")).strip()
        generic = str(ext.get("generic_name", "")).strip()
        conf = float(ext.get("confidence", 0.0))
        if name and conf > best_name_conf:
            best_name = name
            best_name_conf = conf
        if generic and conf > best_generic_conf:
            best_generic = generic
            best_generic_conf = conf
    overall_confidence = sum(
        float(ext.get("confidence", 0.0)) for ext in extractions
    ) / len(extractions)

    return {
        "medicine_name": medicine_name,
        "name": best_name,
        "generic_name": best_generic,
        "confidence": round(overall_confidence, 3),
    }


def extract_regulatory(chunks: list[dict]) -> Any:
    extractions: list[dict] = []
    medicine_name = (chunks[0].get("medicine_name") if chunks else None) or "unknown"

    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        try:
            resp = _extract_with_llm(
                human_content=text,
                system_content=regulatory_prompt,
                structure=Regulatory,
            )
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    continue
            if isinstance(resp, dict):
                extractions.append(resp)
        except Exception as e:
            print(f"Regulatory extraction error: {e}")
            continue

    if not extractions:
        return {
            "medicine_name": medicine_name,
            "banned": None,
            "prescription_required": None,
            "confidence": 0.0,
        }

    # Resolve booleans conservatively: any True wins; explicit False wins over None
    banned = None
    rx = None

    for e in extractions:
        b = e.get("banned")
        if b is True:
            banned = True
        elif b is False and banned is None:
            banned = False

        r = e.get("prescription_required")
        if r is True:
            rx = True
        elif r is False and rx is None:
            rx = False

    overall_conf = sum(float(e.get("confidence", 0.0)) for e in extractions) / len(
        extractions
    )

    return {
        "medicine_name": medicine_name,
        "banned": banned,
        "prescription_required": rx,
        "confidence": round(overall_conf, 3),
    }


def merge_fields(fields: dict) -> Any:
    pass


def validate(schema: dict) -> Any:
    pass


def save_to_postgres(kv: dict):
    conninfo = (
        f"host={os.getenv('PGHOST')} "
        f"port={os.getenv('PGPORT', 5432)} "
        f"user={os.getenv('PGUSER')} "
        f"password={os.getenv('PGPASSWORD')} "
        f"dbname={os.getenv('PGDATABASE')}"
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            for name, schema in kv.items():
                cur.execute(
                    """
                    INSERT INTO new_medicines (medicine_name, data)
                    VALUES (%s, %s)
                    ON CONFLICT (medicine_name)
                    DO UPDATE SET data = new_medicines.data || EXCLUDED.data
                    """,
                    (name, json.dumps(schema)),
                )
        conn.commit()
