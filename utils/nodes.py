import base64
import json
from io import BytesIO
from itertools import combinations
from operator import add
from os import stat
from re import L
from typing import Annotated, List, Literal, Tuple, TypedDict

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
from utils.states import Decision, MedicineList, OutputState, OverallState, UserProfile
from utils.tools import (
    get_combinations_cache,
    get_image_hash,
    get_single_names_cache,
    get_vector_db,
    normalise_drug_name,
    normalise_drug_names,
    set_image_hash,
    write_combinations_cache,
    write_single_names_cache,
)

load_dotenv()

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
groq_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)


def validate_input(state: OverallState) -> Command[Literal[END, "check_image_quality"]]:
    user = state["user"]
    if not user["name"] or not user["age"] or not user["gender"]:
        return Command(goto=END, update={"isValid": False})
    if not state["image_base64"] or not state["image_mime"]:
        return Command(goto=END, update={"isValid": False})

    return Command(goto="check_image_quality", update={"isValid": True})


def extract_image(state: OverallState) -> Command[Literal["human_approval"]]:
    image_url = f"data:{state['image_mime']};base64,{state['image_base64']}"

    res = get_image_hash(state["image_hash"], model=MedicineList)
    if res:
        return Command(goto="human_approval", update={"medicinelist": res})

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
    set_image_hash(state["image_hash"], response)

    return Command(goto="human_approval", update={"medicinelist": response})


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
# 1. check dosage (?)
# 2. move the prompt from this function to prompt file
def ban_check(
    state: OverallState,
) -> Command[Literal["sql_graph", "nlem_check", "generate_explanation"]]:
    medicines = state.get("medicinelist", [])
    if not medicines:
        return Command(
            goto="nlem_check", update={"decision": Decision.PASS, "ban_hits": []}
        )

    normalised_medicines = [normalise_drug_names(m) for m in medicines.medicinelist]
    single_names = get_single_names_cache(key="banneddrugs")
    plus_combinations = get_combinations_cache(key="plus_combinations")
    fixed_combinations = get_combinations_cache(key="fixed_dose")

    lst = [single_names, plus_combinations, fixed_combinations]

    for ele in lst:
        if not ele:
            return Command(goto="sql_graph")

    ban_hits = []

    for med in normalised_medicines:
        if med in single_names:
            ban_hits.append(med)

    if ban_hits:
        return Command(
            goto="generate_explanation",
            update={"decision": Decision.FAIL, "ban_hits": ban_hits},
        )

    results = []
    for i in range(1, len(normalised_medicines) + 1):
        results.extend(combinations(normalised_medicines, i))

    for comb in results:
        if comb in plus_combinations:
            ban_hits.append(comb)

    if ban_hits:
        return Command(
            goto="generate_explanation",
            update={"decision": Decision.FAIL, "ban_hits": ban_hits},
        )

    llm_prompt = f"""
    You are a medical rule-matching assistant.
    Input:
    - medicines: {json.dumps(normalised_medicines)}
    - banned_rules: {fixed_combinations}
    Task:
    - For each banned rule, determine whether it applies to the given medicines.
    - A rule applies if the medicines satisfy the medical or pharmacological meaning of the rule.
    - Use standard medical knowledge (e.g., drug classes, interactions, synonyms).
    Output Rules:
    - Return ONLY a JSON list of banned rules that apply.
    - If none apply, return [].
    - Do NOT explain your reasoning.
    - Do NOT invent medicines not present in the input list.
    - Be conservative: if unsure, do NOT include the rule.
    """

    response = groq_model.invoke(llm_prompt)

    if response.content:
        ban_hits.append(response.content)

    if ban_hits:
        return Command(
            goto="generate_explanation",
            update={"decision": Decision.FAIL, "ban_hits": ban_hits},
        )

    return Command(
        goto="nlem_check", update={"decision": Decision.PASS, "ban_hits": []}
    )


# TODO: move the prompt from this function to prompt file
def nlem_check(state: OverallState) -> Command[Literal["generate_explanation"]]:
    medicines = state.get("medicinelist", [])
    if not medicines:
        return Command(
            goto="generate_explanation",
            update={"decision": Decision.WARN, "nlem_hits": [], "nlem_misses": []},
        )

    vectorstore = get_vector_db(collection_name="drugs")

    nlem_hits, nlem_misses = [], []
    normalised_medicines = [normalise_drug_names(m) for m in medicines]

    candidates = {}

    for med in normalised_medicines:
        docs = vectorstore.similarity_search(
            query=med,
            k=5,
            filter={"source": "data/nlem2022.pdf"},
        )

        candidate_names = [normalise_drug_name(doc.page_content) for doc in docs]
        candidates[med] = candidate_names

    llm_prompt = f"""
You are an expert medical terminology normalization assistant specializing in drug name matching.

Input:
- A dictionary where each key is a raw drug mention (candidate) extracted from text.
- The value for each key is a list of possible standardized medicine names (e.g., generic names, common synonyms, or known variants) that it might refer to.

Your task:
For each candidate key, determine which (if any) of the provided possible standardized names in its list actually refer to the **same medicine** as the candidate mention.

Use standard pharmacological knowledge, including:
- Generic vs brand name equivalence
- Different salt forms of the same active moiety (e.g., "atorvastatin calcium" ≡ "atorvastatin")
- Common abbreviations and synonyms
- Strength and dosage form differences (ignore these unless they change the active ingredient)
- Regional naming variations (e.g., paracetamol ≡ acetaminophen)

Important rules:
- Only consider a match if the active ingredient(s) are pharmacologically identical.
- Do NOT assume a candidate is a combination drug unless explicitly indicated (e.g., avoid matching "lisinopril" to "lisinopril/HCTZ").
- Do NOT match if only the drug class is the same (e.g., "statin" is not a match for "atorvastatin").
- Ignore differences in strength, dosage form, or manufacturer.
- Be strict and conservative: if there is any ambiguity or uncertainty, do NOT include it.

Output format:
Return ONLY a valid JSON list containing the candidate keys (the raw mentions) that have at least one correct match in their list.

Examples:
- If "Lipitor" has ["atorvastatin", "rosuvastatin"] → include "Lipitor" (matches atorvastatin)
- If "tylenol" has ["paracetamol", "ibuprofen"] → include "tylenol" (matches paracetamol)
- If "blood pressure pill" has ["lisinopril", "metoprolol"] → do NOT include (too vague)
- If none match → return []

Final output must be parseable JSON only. No explanations, no markdown, no extra text.

Candidates:
{json.dumps(candidates)}
"""
    response = groq_model.invoke(llm_prompt)

    if response.content:
        nlem_hits.append(response.content)

    decision = Decision.PASS if not nlem_misses else Decision.WARN

    return Command(
        goto="generate_explanation",
        update={
            "decision": decision,
            "nlem_misses": nlem_misses,
            "nlem_hits": nlem_hits,
        },
    )


# TODO: improve the stucture and prompt of this node to give rightful decision
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
