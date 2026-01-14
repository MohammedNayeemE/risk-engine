import base64
import json
from io import BytesIO
from itertools import combinations
from typing import List, Literal

import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from PIL import Image
from pydantic import BaseModel
from pydantic.type_adapter import P
from sqlalchemy import update

from engine.llm.models import ModelManager
from engine.prompts.risk_prompts import (
    admin_report_prompt,
    check_combination_prompt,
    generate_explanation_prompt,
    image_extract_prompt,
    safety_evaluation_prompt,
)
from engine.states.risk_states import (
    AdminReport,
    Decision,
    MedicineItem,
    MedicineList,
    MedicineSafetyIssue,
    OutputState,
    OverallState,
    SafetyEvaluation,
    UserProfile,
)
from persistence.repositories import (
    get_combinations_cache,
    get_image_hash,
    get_single_names_cache,
    get_vector_db,
    normalise_drug_name,
    normalise_drug_names,
    retrieve_from_sql,
    search_medical_web,
    set_image_hash,
)

load_dotenv()


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
            {"type": "text", "text": image_extract_prompt},
            {
                "type": "image_url",
                "image_url": image_url,
            },
        ]
    )

    gemini_model = ModelManager.get_model_for_image_processing()
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
    if height < 200 or width < 200:
        return Command(
            goto=END,
            update={
                "image_quality": False,
                "image_quality_reason": "Image resolution too low",
            },
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 35:
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


def ban_check(
    state: OverallState,
) -> Command[Literal["sql_graph", "multilayer_sanity_check", "generate_explanation"]]:
    medicines = state.get("medicinelist", [])
    if not medicines:
        return Command(
            goto="multilayer_sanity_check",
            update={"decision": Decision.PASS, "ban_hits": []},
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

    # llm_prompt = f"""
    #     You are a strict medical rule-matching assistant for detecting banned fixed-dose combinations (FDCs).
    #
    #     Input:
    #     - medicines: {json.dumps(normalised_medicines)}
    #     - banned_rules: {fixed_combinations}
    #
    #     Task:
    #     - For each banned rule in banned_rules, check if the prescribed medicines include a single medicine that exactly matches the fixed-dose combination described in the rule.
    #     - A rule applies ONLY if there is one medicine in the input list whose name and composition precisely correspond to the banned fixed-dose combination (e.g., a syrup containing both active ingredients of the banned FDC).
    #     - Do NOT consider separate medicines that together contain the ingredients of a banned rule.
    #     - Do NOT consider pharmacological interactions, contraindications, or any other medical reasoning beyond exact FDC matches.
    #     - Do NOT use any external knowledge, assumptions about ingredients, or generic equivalents.
    #     - Only match based on the medicine names and details explicitly provided in the input medicines list.
    #     - Be extremely conservative: if there is any doubt or if the match is not exact, do NOT include the rule.
    #
    #     Output Rules:
    #     - Return ONLY a JSON list of the banned rules (exactly as provided in banned_rules) that apply.
    #     - If none apply, return [].
    #     - Do NOT include any explanation, reasoning, or additional text.
    #     """
    #
    # response = groq_model.invoke(llm_prompt)
    #
    # if response.content:
    #     ban_hits.append(response.content)
    #
    # if ban_hits:
    #     return Command(
    #         goto="generate_explanation",
    #         update={"decision": Decision.WARN, "ban_hits": ban_hits},
    #     )

    return Command(
        goto="multilayer_sanity_check",
        update={"decision": Decision.PASS, "ban_hits": []},
    )


required_fields = ["name", "forms", "strength", "dosage", "age", "regulatory"]


def find_missing_fields(medicine_data: dict, required_fields: list[str]) -> list[str]:
    missing = []

    for field in required_fields:
        value = medicine_data.get(field)
        if value is None or value == "" or value == []:
            missing.append(field)

    return missing


class MedicineSchema(BaseModel):
    name: str
    generic_name: str
    forms: Literal["tablet", "liquid", "injection"]
    strength_mg: str
    dosage_limit: str
    age_restrictions: str
    regulatory: str


def evaluate_medicine_safety(
    medicine_item: MedicineItem,
    medicine_data: dict,
    user_profile: UserProfile,
    other_medicines: list[MedicineItem],
) -> SafetyEvaluation:
    """
    Evaluate the safety of a medicine for a specific patient.

    Args:
        medicine_item: The prescribed medicine
        medicine_data: Database information about the medicine
        user_profile: Patient's profile (age, allergies, conditions)
        other_medicines: List of other medicines in prescription (for interaction check)

    Returns:
        SafetyEvaluation with decision and detailed issues
    """

    # Build context for LLM evaluation
    patient_context = f"""
Patient Profile:
- Age: {user_profile.get('age')}
- Gender: {user_profile.get('gender')}
- Allergies: {', '.join(user_profile.get('allergies', [])) or 'None'}
- Known Conditions: {', '.join(user_profile.get('known_conditions', [])) or 'None'}
"""

    medicine_context = f"""
Prescribed Medicine:
- Name: {medicine_item.name}
- Form: {medicine_item.form}
- Strength: {medicine_item.strength}
- Dose: {medicine_item.dose}
- Dosage: {medicine_item.dosage}

Database Information:
{json.dumps(medicine_data, indent=2)}

Other Medicines in Prescription:
{', '.join([m.name for m in other_medicines if m.name and m.name != medicine_item.name]) or 'None'}
"""

    evaluation_prompt = f"{patient_context}\n{medicine_context}"

    # Get LLM evaluation
    messages = [
        SystemMessage(content=safety_evaluation_prompt),
        HumanMessage(content=evaluation_prompt),
    ]

    groq_model = ModelManager.get_model_for_text_extraction()
    response = groq_model.invoke(messages)

    # Parse LLM response
    try:
        response_text = response.content.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        evaluation_data = json.loads(response_text)

        # Build MedicineSafetyIssue objects
        issues = []
        for issue_data in evaluation_data.get("issues", []):
            issues.append(
                MedicineSafetyIssue(
                    medicine_name=medicine_item.name,
                    issue_type=issue_data.get("issue_type", "other"),
                    severity=issue_data.get("severity", "info"),
                    description=issue_data.get("description", ""),
                    recommendation=issue_data.get("recommendation"),
                )
            )

        # Determine overall decision based on issues
        decision = Decision.PASS
        if any(issue.severity == "critical" for issue in issues):
            decision = Decision.FAIL
        elif any(issue.severity == "warning" for issue in issues):
            decision = Decision.WARN

        return SafetyEvaluation(
            medicine_name=medicine_item.name,
            decision=decision,
            issues=issues,
            age_appropriate=evaluation_data.get("age_appropriate", True),
            dosage_safe=evaluation_data.get("dosage_safe", True),
            allergy_safe=evaluation_data.get("allergy_safe", True),
            interaction_safe=evaluation_data.get("interaction_safe", True),
            condition_compatible=evaluation_data.get("condition_compatible", True),
        )

    except json.JSONDecodeError as e:
        print(f"Error parsing safety evaluation response: {e}")
        # Fallback: create a basic evaluation with warning
        return SafetyEvaluation(
            medicine_name=medicine_item.name,
            decision=Decision.WARN,
            issues=[
                MedicineSafetyIssue(
                    medicine_name=medicine_item.name,
                    issue_type="other",
                    severity="warning",
                    description="Unable to complete full safety evaluation. Requires manual review.",
                    recommendation="Manual review by pharmacist recommended",
                )
            ],
        )


def multilayer_sanity_check(
    state: OverallState,
) -> Command[Literal["generate_explanation", END]]:
    """
    Multi-layer sanity check that:
    1. Fetches medicine data from database
    2. Identifies missing fields
    3. Searches web for missing information
    4. Evaluates safety for each medicine
    5. Makes PASS/WARN/FAIL decision
    """

    # Extract medicines from state
    medicine_list_wrapper: MedicineList = state.get("medicinelist")
    if not medicine_list_wrapper or not medicine_list_wrapper.medicinelist:
        medicines = []
    else:
        medicines = medicine_list_wrapper.medicinelist  # List[MedicineItem]

    # Step 1: Fetch medicine data from database
    data_from_knowledge_base = retrieve_from_sql(medicine_list_wrapper)

    if not data_from_knowledge_base:
        return Command(
            update={
                "decision": Decision.WARN,
                "db_medicine_data": {},
            },
            goto="generate_explanation",
        )

    # Step 2: Identify missing medicines and fields
    db_names_lower = {name.lower(): name for name in data_from_knowledge_base.keys()}
    prescription_names_lower = {med.name.lower() for med in medicines if med.name}

    missing_medicines = prescription_names_lower - set(db_names_lower.keys())
    common_names_lower = prescription_names_lower & set(db_names_lower.keys())

    missing_medicines_original = [
        med.name
        for med in medicines
        if med.name and med.name.lower() in missing_medicines
    ]

    missing_fields_per_medicine = {}
    for med in medicines:
        if not med.name or med.name.lower() not in common_names_lower:
            continue
        original_db_name = db_names_lower[med.name.lower()]
        med_data = data_from_knowledge_base[original_db_name]
        missing_fields = find_missing_fields(med_data, required_fields)
        if missing_fields:
            missing_fields_per_medicine[med.name] = missing_fields

    # Step 3: Search web for missing fields
    web_search_results = {}
    if missing_fields_per_medicine:
        for medicine_name, missing_fields in missing_fields_per_medicine.items():
            search_query = f"{medicine_name} appropriate details dosage age restrictions safety India {' '.join(missing_fields)}"
            web_search_results[medicine_name] = search_medical_web(search_query)

    # Step 4: Merge web search results into medicine data
    for medicine_name, search_result in web_search_results.items():
        if search_result.get("answer"):
            if medicine_name not in data_from_knowledge_base:
                data_from_knowledge_base[medicine_name] = {}
            # Add web search answer to medicine data
            data_from_knowledge_base[medicine_name]["web_search_info"] = search_result[
                "answer"
            ]

    # Step 5: Evaluate safety for each medicine
    safety_evaluations = []
    all_issues = []
    critical_issues = []
    warning_issues = []

    user_profile = state.get("user", {})

    for med in medicines:
        if not med.name:
            continue

        # Get medicine data (from DB or web)
        med_data = (
            data_from_knowledge_base.get(med.name.lower())
            or data_from_knowledge_base.get(med.name)
            or {}
        )

        # Evaluate safety
        evaluation = evaluate_medicine_safety(med, med_data, user_profile, medicines)
        safety_evaluations.append(evaluation)
        all_issues.extend(evaluation.issues)

        if any(issue.severity == "critical" for issue in evaluation.issues):
            critical_issues.extend(
                [issue for issue in evaluation.issues if issue.severity == "critical"]
            )
        if any(issue.severity == "warning" for issue in evaluation.issues):
            warning_issues.extend(
                [issue for issue in evaluation.issues if issue.severity == "warning"]
            )

    # Step 6: Determine overall decision
    overall_decision = Decision.PASS

    if critical_issues:
        overall_decision = Decision.FAIL
    elif warning_issues or missing_medicines_original:
        overall_decision = Decision.WARN

    # Update state with all collected information
    return Command(
        update={
            "missing_fields": missing_fields_per_medicine,
            "missing_medicines": missing_medicines_original,
            "web_search_results": web_search_results,
            "db_medicine_data": data_from_knowledge_base,
            "safety_evaluations": safety_evaluations,
            "decision": overall_decision,
            "safety_issues": all_issues,
        },
        goto="generate_explanation",
    )


# TODO: improve the stucture and prompt of this node to give rightful decision
# improve the scroing system of this node , instead of just random system_prompt , generate explanation , report and summary
# save and send the report and summary to the admin via mail agent
def generate_explanation(state: OverallState) -> Command[Literal[END]]:
    """
    Generate comprehensive explanation and admin report based on decision.

    Handles three decision types:
    - FAIL: Medicine is contraindicated or unsafe
    - WARN: Requires admin/pharmacist review
    - PASS: Safe to dispense
    """

    decision = state.get("decision", Decision.PASS)
    user_profile = state.get("user", {})
    safety_evaluations = state.get("safety_evaluations", [])
    ban_hits = state.get("ban_hits", [])
    safety_issues = state.get("safety_issues", [])

    # Separate issues by severity
    critical_issues = [issue for issue in safety_issues if issue.severity == "critical"]
    warning_issues = [issue for issue in safety_issues if issue.severity == "warning"]
    info_issues = [issue for issue in safety_issues if issue.severity == "info"]

    # Extract medicine names from evaluations
    medicines_evaluated = [eval.medicine_name for eval in safety_evaluations]

    # Generate admin report
    admin_report = AdminReport(
        overall_decision=decision,
        patient_age=int(user_profile.get("age", 0)) if user_profile.get("age") else 0,
        patient_gender=user_profile.get("gender", "Not specified"),
        medicines_evaluated=medicines_evaluated,
        safety_evaluations=safety_evaluations,
        banned_findings=ban_hits,
        critical_issues=critical_issues,
        warning_issues=warning_issues,
        requires_approval=(decision == Decision.WARN or decision == Decision.FAIL),
        report_reason=_generate_report_reason(
            decision, ban_hits, critical_issues, warning_issues
        ),
        recommendations=_generate_recommendations(
            decision, critical_issues, warning_issues
        ),
    )

    # Generate explanation message
    explanation_content = _format_explanation_message(
        decision, ban_hits, critical_issues, warning_issues, admin_report
    )

    messages = [
        SystemMessage(content=admin_report_prompt),
        HumanMessage(content=explanation_content),
    ]

    groq_model = ModelManager.get_model_for_text_extraction()
    response = groq_model.invoke(messages)

    return Command(
        goto=END,
        update={
            "admin_report": admin_report,
            "explanation": response,
        },
    )


def _generate_report_reason(
    decision: Decision,
    ban_hits: list[str],
    critical_issues: list[MedicineSafetyIssue],
    warning_issues: list[MedicineSafetyIssue],
) -> str:
    """Generate a concise reason for the decision."""

    if decision == Decision.FAIL:
        if ban_hits:
            return f"Prescription contains banned substances: {', '.join(ban_hits)}"
        elif critical_issues:
            return (
                f"Critical safety issues identified: {critical_issues[0].description}"
            )
        return "Prescription cannot be approved due to safety concerns"

    elif decision == Decision.WARN:
        issues = []
        if ban_hits:
            issues.append(f"Banned substances detected ({len(ban_hits)})")
        if warning_issues:
            issues.append(f"Safety warnings found ({len(warning_issues)})")
        return "Requires admin/pharmacist review: " + "; ".join(issues)

    else:  # PASS
        return "No safety concerns identified. Prescription approved."


def _generate_recommendations(
    decision: Decision,
    critical_issues: list[MedicineSafetyIssue],
    warning_issues: list[MedicineSafetyIssue],
) -> list[str]:
    """Generate actionable recommendations based on decision."""

    recommendations = []

    if decision == Decision.FAIL:
        recommendations.append(
            "REJECT prescription - contains contraindicated medicines"
        )
        recommendations.append("Contact prescribing physician to discuss alternatives")
        if critical_issues:
            for issue in critical_issues:
                if issue.recommendation:
                    recommendations.append(f"- {issue.recommendation}")

    elif decision == Decision.WARN:
        recommendations.append(
            "REQUIRES APPROVAL - Review with pharmacist or medical admin"
        )
        for issue in warning_issues:
            if issue.recommendation:
                recommendations.append(f"- {issue.recommendation}")
        recommendations.append(
            "Consider consulting with prescribing physician if concerns persist"
        )

    else:  # PASS
        recommendations.append("APPROVE prescription - Ready for dispensing")
        recommendations.append("Provide standard patient counseling")

    return recommendations


def _format_explanation_message(
    decision: Decision,
    ban_hits: list[str],
    critical_issues: list[MedicineSafetyIssue],
    warning_issues: list[MedicineSafetyIssue],
    admin_report: AdminReport,
) -> str:
    """Format explanation message for the LLM to create a professional report."""

    message = f"""
Decision: {decision.value.upper()}

Patient Age: {admin_report.patient_age}
Patient Gender: {admin_report.patient_gender}
Medicines Evaluated: {', '.join(admin_report.medicines_evaluated) or 'None'}

Report Reason: {admin_report.report_reason}

Banned Substances Found: {len(ban_hits)} - {', '.join(ban_hits) if ban_hits else 'None'}
Critical Issues: {len(critical_issues)}
Warning Issues: {len(warning_issues)}

Critical Issues Details:
{_format_issues(critical_issues) if critical_issues else 'None'}

Warning Issues Details:
{_format_issues(warning_issues) if warning_issues else 'None'}

Recommendations:
{chr(10).join(f'- {rec}' for rec in admin_report.recommendations)}

Please generate a professional medical report based on this information, suitable for admin/pharmacist review.
Include clear decision statement, reasons, and next steps.
"""
    return message


def _format_issues(issues: list[MedicineSafetyIssue]) -> str:
    """Format safety issues for display."""
    formatted = []
    for issue in issues:
        formatted.append(
            f"- [{issue.severity.upper()}] {issue.medicine_name}: {issue.description}"
        )
        if issue.recommendation:
            formatted.append(f"  Recommendation: {issue.recommendation}")
    return "\n".join(formatted)
