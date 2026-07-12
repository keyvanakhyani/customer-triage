"""The core triage chains: classify, extract, and draft replies."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
            "system",
            "You extract structured data from a customer message. "
            "Return a JSON object with these keys: "
            "name (string or null), "
            "order_number (string or null), "
            "urgency (one of: low, medium, high), "
            "language (ISO code like 'en' or 'fa'). "
            "Respond with ONLY the JSON object, no explanation.",
        ),
        ("user", "{message}"),
    ])

    model = get_model()
    parser = JsonOutputParser()

    return prompt | model | parser


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