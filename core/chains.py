"""LCEL chain builders. Each chain is prompt | model | parser."""

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from core.config import get_model
from core.prompts import GUARDRAILS, build_classify_prompt


def build_classify_chain():
    """Classify a message. Returns the category as a plain string."""
    return build_classify_prompt() | get_model() | StrOutputParser()

def build_classify_chain_raw():
    """Same chain without the parser, so callers can read response_metadata."""
    return build_classify_prompt() | get_model()


def build_reply_chain():
    """Build a chain that drafts a reply using the message and its context."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful customer service agent for a retail store. "
            f"{GUARDRAILS}\n"
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

