"""
Email service for sending notifications when risk assessment decisions are WARN.
"""

import logging
from typing import List, Optional

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema

from app.settings import settings

logger = logging.getLogger(__name__)


# Configure email connection
email_config = (
    ConnectionConfig(
        MAIL_USERNAME=settings.MAIL_USERNAME,
        MAIL_PASSWORD=settings.MAIL_PASSWORD,
        MAIL_FROM=settings.MAIL_FROM,
        MAIL_PORT=settings.MAIL_PORT,
        MAIL_SERVER=settings.MAIL_SERVER,
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
    )
    if settings.EMAIL_ENABLED and settings.MAIL_USERNAME
    else None
)


async def send_warn_email(
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    medicines: List[dict],
    issues: List[dict],
    recipient_emails: Optional[List[str]] = None,
) -> bool:
    """
    Send email notification when risk assessment decision is WARN.

    Args:
        patient_name: Name of the patient
        patient_age: Age of the patient
        patient_gender: Gender of the patient
        medicines: List of medicines in the prescription
        issues: List of safety issues found (should be of severity 'warning')
        recipient_emails: List of recipient email addresses (uses WARN_RECIPIENTS from config if not provided)

    Returns:
        bool: True if email sent successfully, False otherwise
    """

    if not settings.EMAIL_ENABLED or not email_config:
        logger.warning("Email service is not enabled or not configured")
        return False

    recipients = recipient_emails or settings.WARN_RECIPIENTS
    if isinstance(recipients, str):
        recipients = recipients.split(",")

    if not recipients:
        logger.warning("No recipient emails configured for WARN decision notifications")
        return False

    try:
        html_content = _build_warn_email_html(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            medicines=medicines,
            issues=issues,
        )

        message = MessageSchema(
            subject=f" Medicine Safety Warning - {patient_name}",
            recipients=recipients,
            body=html_content,
            subtype="html",
        )

        fm = FastMail(email_config)
        await fm.send_message(message)

        logger.info(
            f"WARN notification email sent to {recipients} for patient {patient_name}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to send WARN notification email: {str(e)}")
        return False


def _build_warn_email_html(
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    medicines: List[dict],
    issues: List[dict],
) -> str:
    """
    Build HTML content for the WARN decision email.
    """

    # Build medicines list HTML
    medicines_html = ""
    if medicines:
        medicines_html = "<ul>"
        for med in medicines:
            if isinstance(med, dict):
                name = med.get("name", "Unknown")
                form = med.get("form", "")
                strength = med.get("strength", "")
                dose = med.get("dose", "")
                dosage = med.get("dosage", "")

                med_str = f"{name}"
                if form:
                    med_str += f" ({form})"
                if strength:
                    med_str += f" {strength}"
                if dosage:
                    med_str += f" - {dosage}"
                medicines_html += f"<li>{med_str}</li>"
            else:
                medicines_html += f"<li>{str(med)}</li>"
        medicines_html += "</ul>"

    # Build issues list HTML
    issues_html = ""
    if issues:
        issues_html = "<ul>"
        for issue in issues:
            if isinstance(issue, dict):
                issue_type = issue.get("issue_type", "Unknown")
                description = issue.get("description", "")
                recommendation = issue.get("recommendation", "")

                issue_str = f"<strong>{issue_type}</strong>: {description}"
                if recommendation:
                    issue_str += f" <em>Recommendation: {recommendation}</em>"
                issues_html += f"<li>{issue_str}</li>"
            else:
                issues_html += f"<li>{str(issue)}</li>"
        issues_html += "</ul>"

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            .header {{
                background-color: #ff9800;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .section {{
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-left: 4px solid #ff9800;
            }}
            .section h3 {{
                color: #ff9800;
                margin-top: 0;
            }}
            .footer {{
                font-size: 12px;
                color: #666;
                margin-top: 30px;
                border-top: 1px solid #ddd;
                padding-top: 10px;
            }}
            ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            li {{
                margin: 8px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>⚠️ Medicine Safety Warning Alert</h2>
        </div>
        
        <div class="section">
            <h3>Patient Information</h3>
            <p><strong>Name:</strong> {patient_name}</p>
            <p><strong>Age:</strong> {patient_age}</p>
            <p><strong>Gender:</strong> {patient_gender}</p>
        </div>
        
        <div class="section">
            <h3>Prescribed Medicines</h3>
            {medicines_html if medicines_html else "<p>No medicines information available.</p>"}
        </div>
        
        <div class="section">
            <h3>⚠️ Safety Warnings Found</h3>
            {issues_html if issues_html else "<p>No specific issues documented.</p>"}
        </div>
        
        <div class="section">
            <h3>Required Action</h3>
            <p>A pharmacist or healthcare professional should review this prescription before dispensing.</p>
            <p>Please consult with the patient and prescribing physician regarding the identified safety concerns.</p>
        </div>
        
        <div class="footer">
            <p>This is an automated alert from the Risk Engine API. Please do not reply to this email.</p>
            <p>For support, contact your system administrator.</p>
        </div>
    </body>
    </html>
    """

    return html
