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
    """Primary model, with automatic fallbacks if it fails."""

    def _build(name: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    primary = _build("google/gemma-4-26b-a4b-it:free").with_retry(stop_after_attempt=3)
    backups = [_build("openai/gpt-oss-120b:free"), _build("openai/gpt-oss-20b:free"),]
    return primary.with_fallbacks(backups)