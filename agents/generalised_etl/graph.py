from langgraph.graph import END, START, StateGraph

from agents.generalised_etl.nodes import (
    age_agent,
    agentic_chunker,
    dosage_agent,
    identity_agent,
    load_documents,
    merge_schema,
    normalise_text,
    regulatory_agent,
    route_chunks,
    save_to_database,
    strength_agent,
    validate_schema,
)
from agents.generalised_etl.states import IngestionState

builder = StateGraph(IngestionState)

builder.add_node("load_documents", load_documents)
builder.add_node("normalise_text", normalise_text)
builder.add_node("agentic_chunker", agentic_chunker)
builder.add_node("route_chunks", route_chunks)
builder.add_node("identity_agent", identity_agent)
builder.add_node("strength_agent", strength_agent)
builder.add_node("dosage_agent", dosage_agent)
builder.add_node("age_agent", age_agent)
builder.add_node("regulatory_agent", regulatory_agent)
builder.add_node("merge_schema", merge_schema)
builder.add_node("validate_schema", validate_schema)
builder.add_node("save_to_database", save_to_database)

builder.add_edge(START, "load_documents")
builder.add_edge("load_documents", "normalise_text")
builder.add_edge("normalise_text", "agentic_chunker")
builder.add_edge("agentic_chunker", "route_chunks")

builder.add_edge("identity_agent", "merge_schema")
builder.add_edge("strength_agent", "merge_schema")
builder.add_edge("dosage_agent", "merge_schema")
builder.add_edge("age_agent", "merge_schema")
builder.add_edge("regulatory_agent", "merge_schema")

builder.add_edge("merge_schema", "validate_schema")
builder.add_edge("validate_schema", "save_to_database")
builder.add_edge("save_to_database", END)

etl = builder.compile()
