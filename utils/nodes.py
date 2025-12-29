import base64
from io import BytesIO
from operator import add
from os import stat
from re import L
from typing import Annotated, List, Literal, TypedDict

import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from PIL import Image
from pydantic import BaseModel
from pydantic.type_adapter import P
from sqlalchemy import update

from utils.prompts import SYSTEM_PROMPTS
from utils.states import (
    IMAGE_HASH_SET,
    Decision,
    MedicineList,
    OutputState,
    OverallState,
    UserProfile,
)
from utils.tools import get_vector_db, normalise_drug_name, normalise_drug_names

load_dotenv()

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
groq_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)


def validate_input(state: OverallState) -> Command[Literal[END, "check_image_quality"]]:
    user = state["user"]
    if not user["name"] or not user["age"] or not user["gender"]:
        return Command(goto=END, update={"isValid": False})
    if state["image_hash"] in IMAGE_HASH_SET:
        return Command(goto=END, update={"isValid": False})
    if not state["image_base64"] or not state["image_mime"]:
        return Command(goto=END, update={"isValid": False})

    return Command(goto="check_image_quality", update={"isValid": True})


def extract_image(state: OverallState) -> Command[Literal["human_approval"]]:
    image_url = f"data:{state['image_mime']};base64,{state['image_base64']}"

    message = HumanMessage(
        content=[
            {"type": "text", "text": SYSTEM_PROMPTS["image_extract_prompt"]},
            {
                "type": "image_url",
                "image_url": image_url,
            },
        ]
    )

    response = gemini_model.with_structured_output(MedicineList).invoke([message])
    extracted = response.medicinelist

    return Command(goto="human_approval", update={"medicinelist": extracted})


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
) -> Command[Literal["ban_check", "extract_image"]]:
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
        return Command(goto="ban_check")

    if action == "regenerate":
        return Command(goto="extract_image")

    if action == "edit":
        updated = decision.get("medicinelist", current)
        return Command(goto="ban_check", update={"medicinelist": updated})

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


# TODO: use multilayer checks
# include the check with 1. exact name -> 2. combination -> 3.dosage
def ban_check(
    state: OverallState,
) -> Command[Literal["nlem_check", "generate_explanation"]]:
    medicines = state.get("medicinelist", [])
    if not medicines:
        return Command(
            goto="generate_explanation",
            update={"decison": Decision.FAIL, "ban_hits": []},
        )

    normalised_medicines = [normalise_drug_names(m) for m in medicines]

    ban_hits = []
    vectorstore = get_vector_db(collection_name="drugs")

    for med in normalised_medicines:

        docs = vectorstore.similarity_search(
            query=med, k=5, filter={"source": "banneddrugs.pdf"}
        )

        for doc in docs:
            candidate = normalise_drug_name(doc.page_content)
            if candidate == med:
                ban_hits.append(doc.metadata)

    if ban_hits:
        return Command(
            goto="generate_explanation",
            update={"decison": Decision.FAIL, "ban_hits": ban_hits},
        )

    return Command(goto="nlem_check", update={"decison": Decision.PASS, "ban_hits": []})


# TODO: prolly use llm here to find subsets
def nlem_check(state: OverallState) -> Command[Literal["generate_explanation"]]:
    medicines = state.get("medicinelist", [])
    if not medicines:
        return Command(
            goto="generate_explanation",
            update={"decison": Decision.WARN, "nlem_hits": [], "nlem_misses": []},
        )

    vectorstore = get_vector_db(collection_name="drugs")

    nlem_hits, nlem_misses = [], []

    normalised_medicines = [normalise_drug_names(m) for m in medicines]

    for med in normalised_medicines:
        docs = vectorstore.similarity_search(
            query=med, k=5, filter={"source": "data/nlem2022.pdf"}
        )
        matched = False
        for doc in docs:
            med_name = normalise_drug_name(doc.page_content)

            if med == med_name:
                matched = True
                nlem_hits.append(doc.metadata)

        if not matched:
            nlem_misses.append(med)

    decision = Decision.PASS if not nlem_misses else Decision.WARN

    return Command(
        goto="generate_explanation",
        update={
            "decision": decision,
            "nlem_misses": nlem_misses,
            "nlem_hits": nlem_hits,
        },
    )


def generate_explanation(state: OverallState) -> Command[Literal[END]]:
    context = {
        "decision": state.get("decision", []),
        "banned_findings": state.get("ban_hits", []),
        "nlem_findings": state.get("nlem_hits", []),
    }
    system_prompt = SYSTEM_PROMPTS["generate_explanation_prompt"]
    user_content = f"""
    Based on the following analysis, generate a clear and professional explanation:

    Decision: {context['decision']}
    Banned substance findings: {context['banned_findings']}
    Non-listed entity matches (NLEM): {context['nlem_findings']}

    Provide a concise, neutral, and factual explanation suitable for reporting.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = groq_model.invoke(messages)

    return Command(goto=END, update={"explanation": response})
