import json
from collections import defaultdict
from typing import Any, List, Literal

from langgraph.graph import END
from langgraph.types import Command, Send

from agents.generalised_etl.constants import AGENT_MAP
from agents.generalised_etl.states import IngestionState
from agents.generalised_etl.tools import (
    FileLoader,
    extract_age,
    extract_dosage,
    extract_identity,
    extract_regulatory,
    extract_strength,
    llm_chunk,
    merge_fields,
    normalize_docs,
    save_to_postgres,
    validate,
)


def load_documents(state: IngestionState) -> Command[Literal["normalise_text", END]]:
    docs = FileLoader.load(state["input_path"])
    goto_node = END if docs == None else "normalise_text"
    return Command(
        update={
            "raw_documents": docs,
        },
        goto=goto_node,
    )


def normalise_text(state: IngestionState) -> Command[Literal["agentic_chunker"]]:
    text = normalize_docs(state["raw_documents"])
    return Command(
        update={
            "normalised_text": text,
        },
        goto="agentic_chunker",
    )


def agentic_chunker(state: IngestionState) -> Command[Literal["route_chunks"]]:
    chunks = llm_chunk(state["normalised_text"])
    return Command(update={"chunks": chunks}, goto="route_chunks")


def route_chunks(state: IngestionState) -> Command:
    if not state["chunks"]:
        return Command(goto=END)
    sends = []
    medicine_chunks = defaultdict(list)

    for chunk in state["chunks"]:
        med_name = chunk.get("medicine_name", "unknown").lower().strip() or "unknown"
        medicine_chunks[med_name].append(chunk)

    for med_name, med_chunks in medicine_chunks.items():
        med_state = {
            **state,
            "chunks": med_chunks,
            "medicine_name": med_name.capitalize(),
        }

        for agent_type, agent_name in AGENT_MAP.items():
            relevant_chunks = [c for c in med_chunks if c["chunk_type"] == agent_type]
            if relevant_chunks:
                agent_state = {**med_state, "chunks": relevant_chunks}
                sends.append(Send(agent_name, agent_state))

    if not sends:
        return Command(goto=END)

    return Command(goto=sends)


def identity_agent(state: IngestionState) -> Command[Literal["merge_schema"]]:
    med = state["medicine_name"]

    result = extract_identity(state["chunks"], med)

    return Command(
        update={
            "extracted_fields": {
                med: {
                    **state.get("extracted_fields", {}).get(med, {}),
                    "identity": result,
                }
            }
        },
        goto="merge_schema",
    )


def strength_agent(state: IngestionState) -> Command[Literal["merge_schema"]]:
    med = state["medicine_name"]

    result = extract_strength(state["chunks"], med)

    return Command(
        update={
            "extracted_fields": {
                med: {
                    **state.get("extracted_fields", {}).get(med, {}),
                    "strength": result,
                }
            }
        },
        goto="merge_schema",
    )


def dosage_agent(state: IngestionState) -> Command[Literal["merge_schema"]]:
    med = state["medicine_name"]
    result = extract_dosage(state["chunks"])
    return Command(
        update={
            "extracted_fields": {
                med: {
                    **state.get("extracted_fields", {}).get(med, {}),
                    "dosage": result,
                }
            }
        },
        goto="merge_schema",
    )


def age_agent(state: IngestionState) -> Command[Literal["merge_schema"]]:
    med = state["medicine_name"]
    result = extract_age(state["chunks"])
    return Command(
        update={
            "extracted_fields": {
                med: {
                    **state.get("extracted_fields", {}).get(med, {}),
                    "age": result,
                }
            }
        },
        goto="merge_schema",
    )


def regulatory_agent(state: IngestionState) -> Command[Literal["merge_schema"]]:
    med = state["medicine_name"]
    result = extract_regulatory(state["chunks"])
    return Command(
        update={
            "extracted_fields": {
                med: {
                    **state.get("extracted_fields", {}).get(med, {}),
                    "regulatory": result,
                }
            }
        },
        goto="merge_schema",
    )


def merge_schema(state: IngestionState) -> Command[Literal["validate_schema"]]:

    return Command(
        update={"medicine_schemas": state["extracted_fields"]},
        goto="validate_schema",
    )


def validate_schema(state: IngestionState) -> Command[Literal[END]]:

    return Command(goto=END)


def save_to_database(state: IngestionState) -> Command[Literal[END]]:

    save_to_postgres(state["medicine_schemas"])

    return Command(goto=END)
