import base64
import hashlib

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from engine.graphs.sql_graph import sql_graph
from engine.nodes.risk_nodes import (
    ban_check,
    check_image_quality,
    extract_image,
    generate_explanation,
    human_approval,
    multilayer_sanity_check,
    validate_input,
)
from engine.states.risk_states import InputState, OutputState, OverallState, UserProfile

load_dotenv()

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)


builder.add_node("validate_input", validate_input)
builder.add_node("check_image_quality", check_image_quality)
builder.add_node("extract_image", extract_image)
builder.add_node("human_approval", human_approval)
builder.add_node("ban_check", ban_check)
builder.add_node("multilayer_sanity_check", multilayer_sanity_check)
builder.add_node("generate_explanation", generate_explanation)
builder.add_node("sql_graph", sql_graph)

builder.add_edge(START, "validate_input")
# builder.add_conditional_edges(
#     "ban_check",
#     route,
#     {
#         "sql_graph": "sql_graph",
#         "multilayer_sanity_check": "multilayer_sanity_check",
#         "generate_explanation": "generate_explanation",
#     },
# )
builder.add_edge("sql_graph", "ban_check")

# Initialize checkpointer for human-in-the-loop support
checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])

# TODO: write mock tests


if __name__ == "__main__":

    user = UserProfile(
        name="Nayeem", age="20", gender="M", allergies=[], known_conditions=[]
    )

    with open(
        "/home/nayeem/Downloads/test3.jpg",
        "rb",
    ) as f:
        image_bytes = base64.b64encode(f.read()).decode()

    def compute_image_hash(image_base64: str) -> str:
        image_bytes = base64.b64decode(image_base64)
        return hashlib.sha256(image_bytes).hexdigest()

    image_hash = compute_image_hash(image_bytes)

    graph.invoke(
        {
            "user": user,
            "image_base64": image_bytes,
            "image_mime": "image/jpeg",
            "image_hash": image_hash,
        }
    )
