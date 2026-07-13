"""Load UI text for a given language from the locales folder."""

import json
from pathlib import Path

LOCALES_DIR = Path(__file__).parent.parent / "locales"


def load_locale(lang: str = "en") -> dict:
    """Load the translation dict for the given language code."""
    locale_file = LOCALES_DIR / f"{lang}.json"
    with open(locale_file, encoding="utf-8") as f:
        return json.load(f)