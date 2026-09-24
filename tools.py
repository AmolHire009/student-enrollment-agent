from langchain_core.tools import tool
from data import PROGRAMS, APPLICATIONS


@tool
def get_program_info(program_name: str):
    """Get duration, tuition and prerequisites for a program."""

    program = PROGRAMS.get(program_name)

    if not program:
        return {
            "error": f"Program '{program_name}' was not found."
        }

    return {
        "program_name": program_name,
        "duration": program["duration"],
        "tuition": program["tuition"],
        "prerequisites": program["prerequisites"]
    }


@tool
def check_application_status(applicant_id: str):
    """Check the current status of an application."""

    application = APPLICATIONS.get(applicant_id)

    if not application:
        return {
            "error": f"Application ID '{applicant_id}' was not found."
        }

    return {
        "applicant_name": application["name"],
        "program": application["program"],
        "status": application["status"],
        "next_step": application["next_step"],
        "pending_documents": application["pending_documents"]
    }


@tool
def get_deadlines(program_name: str):
    """Get application and document deadlines for a program."""

    program = PROGRAMS.get(program_name)

    if not program:
        return {
            "error": f"Program '{program_name}' was not found."
        }

    return {
        "program_name": program_name,
        "application_deadline": program["application_deadline"],
        "document_submission_deadline": program["document_deadline"],
        "decision_notification_date": program["decision_date"]
    }
