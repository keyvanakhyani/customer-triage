# Customer Message Triage

An AI-powered tool that reads incoming customer messages, classifies them,
extracts structured data, and drafts a reply — in both Persian and English.

Built with LangChain (LCEL) on OpenRouter's free tier, with a clean architecture
that separates core logic from the interface.

## The problem

Small businesses receive dozens of customer messages daily. Each has to be read,
categorized, checked for an order number, and answered — repetitive work that
eats hours every week.

This tool automates the first pass: it reads the message, decides what kind of
request it is, extracts the details that matter, and drafts a reply in the
customer's language — ready for a human to approve and send.

## Status

🚧 In development. Currently building the core chains.

- [ ] Core: classification chain
- [ ] Core: structured extraction
- [ ] Core: reply drafting
- [ ] i18n: Persian and English locales
- [ ] Gradio web interface
- [ ] Evaluation and accuracy metrics

## Results

Classification accuracy on a bilingual (Persian + English) test set: **75%**.

Error analysis showed two failure modes: ambiguous boundary cases between
categories, and output noise from free-tier models. A resilient output parser
(planned) is expected to recover the noise-related errors.

## Architecture

The core logic is fully separated from the interface, so the "hands" (Gradio
today, Telegram later) are swappable without touching the brain.

```
customer-triage/
├── core/
│   ├── config.py     # central model configuration
│   ├── triage.py     # 3 LCEL chains + orchestrator
│   └── i18n.py       # language file loader
├── locales/          # fa.json / en.json (UI text, separated from code)
├── app_gradio.py     # web interface (RTL-aware)
└── evaluate.py       # accuracy evaluation
```

## Tech stack

- **LangChain (LCEL)** — chain composition (`prompt | model | parser`)
- **OpenRouter** — model access (free tier)
- **Gradio** — web interface
- **Python 3.11**

## Running it

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# add your OpenRouter key
copy .env.example .env
# then edit .env and paste your key

python app_gradio.py
```#