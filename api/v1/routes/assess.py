import base64
import hashlib
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from langgraph.types import Command, StateSnapshot

from api.v1.schemas import (
    ApprovalRequest,
    HumanApprovalResponse,
    ReportResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    UserProfileRequest,
)
from app.middleware.auth import get_current_client
from engine.orchestrator import graph
from engine.states.risk_states import Decision, MedicineItem, MedicineList, UserProfile
from persistence.models import Client
from persistence.repositories import (
    get_client_reports,
    get_thread,
    save_report,
    save_thread,
    update_thread_status,
)
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
    except Exception as exc:
        logger.error("Error computing image hash: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc


ALLOWED_IMAGE_MIME: set[str] = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def detect_image_mime(data: bytes) -> Optional[str]:
    """Detect basic image MIME type from magic bytes.

    Returns a MIME string if recognized, otherwise None.
    """
    if not data or len(data) < 4:
        return None

    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"

    if data[:8] == b"\x89PNG\r\n\x1a\x0a":
        return "image/png"

    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"

    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    if data[:2] == b"BM":
        return "image/bmp"

    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"

    return None


def validate_and_get_image_mime(data: bytes) -> str:
    """Validate that the uploaded file is an image and return its MIME.

    Uses content-based detection (magic bytes). If detection fails, rejects.
    """
    detected = detect_image_mime(data)
    if not detected:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid image (jpeg/png/gif/webp/bmp/tiff)",
        )

    if detected not in ALLOWED_IMAGE_MIME:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    return detected


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
        if (
            state_values.get("admin_report").overall_decision.value
            != Decision.WARN.value
        ):
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
    except Exception as exc:
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
        else:
            raise HTTPException(
                status_code=404,
                detail="The given image is not a valid prescription , please provide a valid prescription",
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
async def approve_medicines(
    approval: ApprovalRequest, client: Client = Depends(get_current_client)
):
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
        state = graph.get_state(config)
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
        try:
            save_report(client_id=client.id or 1, response=response.model_dump())
        except Exception as save_exc:
            logger.error("Failed to save report: %s", save_exc, exc_info=True)

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
        image_mime = validate_and_get_image_mime(image_bytes)
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
            image_mime=image_mime,
        )
        return await start_risk_assessment(request)
    except Exception as e:
        logger.error(f"Error processing file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/get-reports", response_model=list[ReportResponse])
async def get_reports(client: Client = Depends(get_current_client)):
    """
    Retrieve all saved assessment reports for the authenticated client.

    Returns only the explanation summary from each stored assessment.
    Results are ordered by creation date (most recent first).
    """
    try:
        logger.info("Fetching reports for client: %s", client.id)
        reports = get_client_reports(client.id or 1)

        result = []
        for report in reports:
            try:
                response_data = report.response
                explanation_content = ""

                if isinstance(response_data, dict):
                    raw_state = response_data.get("raw_state", {})
                    if isinstance(raw_state, dict):
                        explanation_obj = raw_state.get("explanation", {})
                        if isinstance(explanation_obj, dict):
                            explanation_content = explanation_obj.get("content", "")
                        elif isinstance(explanation_obj, str):
                            explanation_content = explanation_obj

                result.append(
                    ReportResponse(
                        id=report.id or 0,
                        created_at=str(report.created_at) if report.created_at else "",
                        explanation=explanation_content,
                    )
                )
            except Exception as extract_exc:
                logger.warning(
                    "Error extracting explanation from report %s: %s",
                    report.id,
                    extract_exc,
                )
                continue

        return result
    except Exception as exc:
        logger.error("Error fetching reports: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error fetching reports: {str(exc)}"
        )

