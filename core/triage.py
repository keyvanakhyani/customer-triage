"""The core triage chains: classify, extract, and draft replies."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from core.config import get_model

CATEGORIES = ["complaint", "order_inquiry", "product_question", "other"]


def build_classify_chain():
    """Build a chain that classifies a message into one category."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a message classifier for a retail store. "
            "Classify the customer message into exactly ONE of these categories: "
            "complaint, order_inquiry, product_question, other. "
            "Respond with ONLY the category name, nothing else.",
        ),
        ("user", "{message}"),
    ])

    model = get_model()
    parser = StrOutputParser()

    return prompt | model | parser


from langchain_core.output_parsers import JsonOutputParser


def build_extract_chain():
    """Build a chain that extracts structured data from a message."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "You extract structured data from a customer message. "
            "Return a JSON object with these keys: "
            "name (string or null), "
            "order_number (string or null), "
            "urgency (one of: low, medium, high), "
            "language (ISO code like 'en' or 'fa'). "
            "Respond with ONLY the JSON object, no explanation. "
            "Do not add any text before or after the JSON. "
            "Do not add safety labels or prefixes."
        ),
        ("user", "{message}"),
    ])

    model = get_model()
    parser = JsonOutputParser()

    return prompt | model | RunnableLambda(_clean_model_noise) | parser


def build_reply_chain():
    """Build a chain that drafts a reply using the message and its context."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful customer service agent for a retail store. "
            "Write a short, polite reply to the customer's message. "
            "The message category is: {category}. "
            "Known details: {details}. "
            "IMPORTANT: Write your reply ONLY in this language: {language}. "
            "Keep it warm, professional, and under 4 sentences.",
        ),
        ("user", "{message}"),
    ])

    model = get_model()
    parser = StrOutputParser()

    return prompt | model | parser



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



from langchain_core.runnables import RunnableLambda


def _clean_model_noise(message):
    """Strip known noise prefixes that free models sometimes prepend.

    Free models via openrouter/free occasionally emit lines like
    'User Safety: safe' before the real content. This removes them so
    the JSON parser receives clean input.
    """
    text = message.content if hasattr(message, "content") else str(message)

    # Drop any leading lines that are not part of the JSON/answer
    lines = text.strip().split("\n")
    cleaned_lines = [
        line for line in lines
        if not line.strip().lower().startswith("user safety")
    ]

    return "\n".join(cleaned_lines).strip()


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