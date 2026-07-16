"""The core triage chains: classify, extract, and draft replies."""

from core.chains import (
    build_classify_chain,
    build_extract_chain,
    build_reply_chain,
)


def triage_message(message: str) -> dict:
    """Run the full triage pipeline on a single message.

    Returns a dict with the category, extracted details, and draft reply.
    """
    # Step 1: classify
    classify_chain = build_classify_chain()
    category = classify_chain.invoke({"message": message})

    # Step 2: extract structured data
    extract_chain = build_extract_chain()
    details = extract_chain.invoke({"message": message})

    # Step 3: draft a reply using the results of steps 1 and 2
    reply_chain = build_reply_chain()
    reply = reply_chain.invoke({
        "message": message,
        "category": category,
        "details": str(details),
        "language": details.get("language", "en"),
    })

    # Decide whether the team needs an alert, and build one if so
    alert = None
    if _needs_alert(category, details):
        alert = _build_alert(category, details)

    return {
        "category": category,
        "details": details,
        "reply": reply,
        "alert": alert,
    }


def _needs_alert(category: str, details: dict) -> bool:
    """Decide whether this message needs an internal team alert.

    Alert on high urgency or on complaints — cases where a human on the
    team should be notified, not just the customer.
    """
    urgency = details.get("urgency", "low")
    return urgency == "high" or category == "complaint"

def _build_alert(category: str, details: dict) -> str:
    """Build a short internal alert message for the team."""
    name = details.get("name") or "Unknown"
    order = details.get("order_number") or "N/A"
    urgency = details.get("urgency", "low")

    return (
        f"⚠️ ACTION NEEDED — {category} (urgency: {urgency})\n"
        f"Customer: {name} | Order: {order}"
    )