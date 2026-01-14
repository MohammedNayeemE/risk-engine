import base64
import hashlib
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langgraph.types import Command, StateSnapshot

from api.v1.schemas import (
    ApprovalRequest,
    HumanApprovalResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    UserProfileRequest,
)
from engine.orchestrator import graph
from engine.states.risk_states import Decision, MedicineItem, MedicineList, UserProfile
from persistence.repositories import get_thread, save_thread, update_thread_status
from services.notification.email import send_warn_email

router = APIRouter(prefix="/assess", tags=["assessment"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_image_hash(image_base64: str) -> str:
    """Compute SHA256 hash for the prescription image."""
    try:
        return hashlib.sha256(base64.b64decode(image_base64)).hexdigest()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Error computing image hash: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc


def prepare_graph_input(request: RiskAssessmentRequest) -> dict:
    """Prepare state payload expected by the LangGraph executor."""
    user_profile = UserProfile(
        name=request.user_profile.name,
        age=request.user_profile.age,
        gender=request.user_profile.gender,
        allergies=request.user_profile.allergies or [],
        known_conditions=request.user_profile.known_conditions or [],
    )

    image_hash = compute_image_hash(request.image_base64)

    return {
        "user": user_profile,
        "image_base64": request.image_base64,
        "image_mime": request.image_mime,
        "image_hash": image_hash,
    }


def extract_medicines_from_state(medicinelist: MedicineList) -> list[MedicineItem]:
    """Extract medicines from graph state regardless of shape."""
    medicines: list[MedicineItem] = []
    if not medicinelist or not medicinelist.medicinelist:
        return []
    items = medicinelist.medicinelist
    for item in items:
        if isinstance(item, dict):
            medicines.append(MedicineItem(**item))
        else:
            medicines.append(item)
    return medicines


async def send_warn_notification(state: StateSnapshot) -> None:
    """Send WARN notification email when applicable."""
    state_values = state.values
    try:
        if state_values.get("decision") != Decision.WARN:
            return

        user = state_values.get("user", {})
        patient_name = user.get("name", "Unknown")
        patient_age = user.get("age", "Unknown")
        patient_gender = user.get("gender", "Unknown")

        medicines = extract_medicines_from_state(state_values["medicinelist"])
        medicines_data = [
            med.dict() if hasattr(med, "dict") else med for med in medicines
        ]

        safety_issues = state_values.get("safety_issues", [])
        issues_data = [
            {
                "issue_type": (
                    issue.issue_type
                    if hasattr(issue, "issue_type")
                    else issue.get("issue_type", "unknown")
                ),
                "description": (
                    issue.description
                    if hasattr(issue, "description")
                    else issue.get("description", "")
                ),
                "recommendation": (
                    issue.recommendation
                    if hasattr(issue, "recommendation")
                    else issue.get("recommendation")
                ),
            }
            for issue in safety_issues
        ]

        await send_warn_email(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            medicines=medicines_data,
            issues=issues_data,
        )
    except Exception as exc:  # pragma: no cover - side-effect guard
        logger.error("Error sending WARN notification: %s", exc)


def extract_details_from_state(state: StateSnapshot) -> dict:
    state_values = state.values
    user_name = state_values["user"].get("name", "unknown")
    medicines = extract_medicines_from_state(state_values["medicinelist"])
    medicines_found = len(medicines)
    image_hash = state_values["image_hash"]
    is_valid = state_values["isValid"]
    image_quality = state_values["image_quality"]
    web_search_results = {}
    missing_medicines = state_values.get("missing_medicines", [])

    return {
        "user_name": user_name,
        "medicines": medicines,
        "medicines_found": medicines_found,
        "image_hash": image_hash,
        "is_valid": is_valid,
        "image_quality": image_quality,
        "web_search_results": web_search_results,
        "missing_medicines": missing_medicines,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/start", response_model=RiskAssessmentResponse | HumanApprovalResponse)
async def start_risk_assessment(request: RiskAssessmentRequest):
    """Kick off a medicine risk assessment."""
    try:
        logger.info("Starting risk assessment for user: %s", request.user_profile.name)
        graph_input = prepare_graph_input(request)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        logger.info("Invoking LangGraph with thread_id: %s", thread_id)
        result = graph.invoke(graph_input, config)
        state = graph.get_state(config)

        if state.next and "human_approval" in state.next:
            medicines = extract_medicines_from_state(state.values["medicinelist"])
            return HumanApprovalResponse(
                status="awaiting_approval",
                thread_id=thread_id,
                user_name=request.user_profile.name,
                medicines=medicines,
                message="Please review the extracted medicines and approve, regenerate, or edit them.",
                approval_data={
                    "current_node": state.next[0] if state.next else None,
                    "medicines_count": len(medicines),
                },
            )
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error during risk assessment: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing risk assessment: {str(exc)}"
        )


@router.post("/approve", response_model=RiskAssessmentResponse)
async def approve_medicines(approval: ApprovalRequest):
    """Resume a paused assessment after human approval or edits."""
    try:
        logger.info(
            "Processing approval for thread: %s, action: %s",
            approval.thread_id,
            approval.action,
        )

        thread = approval.thread_id
        if not thread:
            raise HTTPException(
                status_code=404,
                detail=f"Thread {approval.thread_id} not found in database",
            )

        config = {"configurable": {"thread_id": approval.thread_id}}
        state = graph.get_state(config)
        if not state.values:
            raise HTTPException(
                status_code=404, detail=f"Thread {approval.thread_id} state not found"
            )
        if approval.action == "approve":
            resume_value: Any = "approve"
        elif approval.action == "regenerate":
            resume_value = "regenerate"
        elif approval.action == "edit":
            if not approval.edited_medicines:
                raise HTTPException(
                    status_code=400,
                    detail="edited_medicines list is required and cannot be empty for 'edit' action",
                )
            resume_value = {
                "action": "edit",
                "medicinelist": [med.dict() for med in approval.edited_medicines],
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Must be 'approve', 'regenerate', or 'edit'",
            )
        result = graph.invoke(Command(resume=resume_value), config=config)
        print(state)
        state_values = extract_details_from_state(state)
        response = RiskAssessmentResponse(
            status="completed",
            user_name=state_values["user_name"],
            medicines_found=state_values["medicines_found"],
            medicines=state_values["medicines"],
            image_hash=state_values["image_hash"],
            is_valid=state_values["is_valid"],
            image_quality=state_values["image_quality"],
            image_quality_reason=state.values.get("image_quality_reason", ""),
            web_search_results=state.values.get("web_search_results", {}),
            missing_medicines=state.values.get("missing_medicines", []),
            missing_fields=state.values.get("missing_fields", {}),
            raw_state=result,
        )

        await send_warn_notification(state)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error processing approval: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing approval: {str(exc)}"
        )


@router.post("/file")
async def assess_risk_file(
    image: UploadFile = File(...),
    name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    allergies: Optional[str] = Form(None),
    known_conditions: Optional[str] = Form(None),
):
    """
    Assess medicine risk from a prescription image (Form/File input).

    Form parameters:
    - image: Prescription image file
    - name: Patient name
    - age: Patient age
    - gender: Patient gender
    - allergies: Comma-separated allergies (optional)
    - known_conditions: Comma-separated conditions (optional)
    """
    try:
        logger.info(f"Processing file upload for user: {name}")
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode()
        allergies_list = [a.strip() for a in allergies.split(",")] if allergies else []
        conditions_list = (
            [c.strip() for c in known_conditions.split(",")] if known_conditions else []
        )
        request = RiskAssessmentRequest(
            user_profile=UserProfileRequest(
                name=name,
                age=age,
                gender=gender,
                allergies=allergies_list,
                known_conditions=conditions_list,
            ),
            image_base64=image_base64,
            image_mime=image.content_type or "image/jpeg",
        )
        return await start_risk_assessment(request)
    except Exception as e:
        logger.error(f"Error processing file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
