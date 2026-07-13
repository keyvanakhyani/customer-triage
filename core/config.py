"""Central configuration: build the LLM once, reuse everywhere."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_model() -> ChatOpenAI:
    """Create and return the configured chat model.

    Reading the model name and key in one place means changing the
    model later is a one-line edit, not a hunt across many files.
    """
    return ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )