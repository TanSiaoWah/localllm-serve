from backend.services.ai_service import analyze_ticket


ticket = {
    "subject": "Cannot log in after password reset",
    "message": (
        "I reset my password this morning but now I can't log in. "
        "It says my password is incorrect even though I just set it. "
        "I need access urgently for a client meeting."
    ),
}

print(analyze_ticket(ticket))