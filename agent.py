import base64
import os

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from utils.nodes import check_image_quality, extract_image, validate_input, human_approval
from utils.states import InputState, OutputState, OverallState, UserProfile

load_dotenv()

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)


builder.add_node("validate_input", validate_input)
builder.add_node("check_image_quality", check_image_quality)
builder.add_node("extract_image", extract_image)
builder.add_node("human_approval", human_approval)

builder.add_edge(START, "validate_input")


graph = builder.compile()

user = UserProfile(
    name="Nayeem", age="20", gender="M", allergies=[], known_conditions=[]
)


with open(
    "/home/nayeem/Downloads/prescription_demo.jpg",
    "rb",
) as f:
    image_bytes = base64.b64encode(f.read()).decode()

graph.invoke(
    {
        "user": user,
        "image_base64": image_bytes,
        "image_mime": "image/jpeg",
    }
)
