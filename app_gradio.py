"""Gradio web interface for the customer triage tool."""

import gradio as gr

from core.triage import triage_message
from core.i18n import load_locale

# Load UI text once. Change "fa" to "en" to switch the whole interface.
t = load_locale("fa")


def process_message(message: str) -> tuple[str, str, str]:
    """Take a message from the UI, run triage, return the three parts."""
    if not message.strip():
        return "—", "—", t["empty_message"]

    result = triage_message(message)

    category = result["category"]
    details = str(result["details"])
    reply = result["reply"]

    return category, details, reply


custom_css = """
.gradio-container {
    direction: rtl;
}
textarea, input {
    text-align: right;
    direction: rtl;
}
label, .gradio-container label span {
    text-align: right !important;
    direction: rtl;
    display: block;
}
.gradio-container p, .gradio-container .prose {
    text-align: right !important;
    direction: rtl;
}
"""

demo = gr.Interface(
    fn=process_message,
    inputs=gr.Textbox(
        lines=4,
        label=t["input_label"],
        placeholder=t["input_placeholder"],
    ),
    outputs=[
         gr.Textbox(label=t["category_label"]),
        gr.Textbox(label=t["details_label"]),
        gr.Textbox(label=t["reply_label"]),
    ],
    title=t["title"],
    description=t["description"],
    css=custom_css,
)

if __name__ == "__main__":
    demo.launch()