"""Gradio web interface for the customer triage tool."""

import gradio as gr

from core.triage import triage_message


def process_message(message: str) -> tuple[str, str, str]:
    """Take a message from the UI, run triage, return the three parts."""
    if not message.strip():
        return "—", "—", "Please enter a message."

    result = triage_message(message)

    category = result["category"]
    details = str(result["details"])
    reply = result["reply"]

    return category, details, reply


demo = gr.Interface(
    fn=process_message,
    inputs=gr.Textbox(
        lines=4,
        label="Customer message",
        placeholder="Paste a customer message here (Persian or English)...",
    ),
    outputs=[
        gr.Textbox(label="Category"),
        gr.Textbox(label="Extracted details"),
        gr.Textbox(label="Draft reply"),
    ],
    title="Customer Message Triage",
    description="Classifies the message, extracts key data, and drafts a reply.",
)

if __name__ == "__main__":
    demo.launch()