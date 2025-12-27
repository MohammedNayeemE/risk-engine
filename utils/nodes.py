import base64
from io import BytesIO
from operator import add
from typing import Annotated, List, Literal, TypedDict

import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from PIL import Image
from pydantic.type_adapter import P

from utils.states import InputState, OutputState, OverallState, UserProfile

load_dotenv()

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def validate_input(state: InputState) -> Command[Literal[END, "check_image_quality"]]:
    user = state["user"]
    if not user["name"] or not user["age"] or not user["gender"]:
        return Command(goto=END, update={"isValid": False})
    if not state["image_base64"] or not state["image_mime"]:
        return Command(goto=END, update={"isValid": False})

    return Command(goto="check_image_quality", update={"isValid": True})


SYS_PROMPT = f"""
You are a medical data extraction assistant.

Your task is to extract medicine names and their corresponding dosages from the given prescription image.

Rules:
- Only extract information that is clearly readable.
- If the image is unclear, incomplete, or medicine names/dosages cannot be confidently identified, respond with exactly:
  Prescription Invalid
- Do not guess or infer missing information.
- Do not include any explanations or extra text.

Output format:
- Return a Python-style list of objects in the following format:

[

  "medicine": "<medicine_name>", "dosage": "<dosage>"
]
"""


def extract_image(state: OverallState) -> Command[Literal["human_approval"]]:
    image_url = f"data:{state['image_mime']};base64,{state['image_base64']}"

    message = HumanMessage(
        content=[
            {"type": "text", "text": SYS_PROMPT},
            {
                "type": "image_url",
                "image_url": image_url,
            },
        ]
    )

    response = gemini_model.invoke([message])

    return Command(goto="human_approval", update={"medicinelist": response.content})


def check_image_quality(state: OverallState) -> Command[Literal["extract_image", END]]:
    try:
        image_bytes = base64.b64decode(state["image_base64"])
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return Command(
            goto=END,
            update={
                "image_quality": False,
                "image_quality_reason": "Invalid or corrupted image file",
            },
        )
    height, width = image.shape[:2]
    if height < 400 or width < 400:
        return Command(
            goto=END,
            update={
                "image_quality": False,
                "image_quality_reason": "Image resolution too low",
            },
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 50:
        return Command(
            goto=END,
            update={
                "image_quality": False,
                "image_quality_reason": "Image is too blurry",
            },
        )
    contrast = gray.std()

    if contrast < 10:
        return Command(
            goto=END,
            update={
                "image_quality": False,
                "image_quality_reason": "Low contrast image",
            },
        )

    return Command(
        goto="extract_image",
        update={"image_quality": True, "image_quality_reason": "OK"},
    )


def human_approval(
    state: OverallState,
) -> Command[Literal[END, "extract_image"]]:
    current = state.get("medicinelist")

    decision = interrupt(
        {
            "type": "human_approval",
            "title": "Review extracted medicine list",
            "message": "Approve, regenerate, or edit the generated medicine list.",
            "medicinelist": current,
            "options": [
                {"action": "approve", "label": "Approve"},
                {"action": "regenerate", "label": "Regenerate"},
                {"action": "edit", "label": "Edit manually"},
            ],
        }
    )

    action = decision.strip().lower()

    if action == "approve":
        return Command(goto=END)

    if action == "regenerate":
        return Command(goto="extract_image")

    if action == "edit":
        updated = decision.get("medicinelist", current)
        return Command(goto=END, update={"medicinelist": updated})

    # fallback → re-interrupt
    return interrupt(
        {
            "type": "human_approval",
            "title": "Review required",
            "message": "Please choose approve, regenerate, or provide an edited list.",
            "medicinelist": current,
            "options": [
                {"action": "approve", "label": "Approve"},
                {"action": "regenerate", "label": "Regenerate"},
                {"action": "edit", "label": "Edit manually"},
            ],
        }
    )
