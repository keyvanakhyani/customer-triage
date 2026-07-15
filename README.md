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

✅ Base project complete.

- [x] Core: classification chain
- [x] Core: structured extraction
- [x] Core: reply drafting
- [x] i18n: Persian and English locales
- [x] Gradio web interface
- [x] Evaluation and accuracy metrics
  
## Roadmap

- [ ] Resilient JSON parser (recover free-model output noise)
- [ ] Alert system for high-urgency messages
- [ ] RAG: ground replies in real product docs

## Results

Classification accuracy on a bilingual (Persian + English) test set:

| Version | Accuracy |
|---|---|
| Baseline | 75% |
| With resilient parser | 87.5%+ |

Error analysis on the baseline showed two failure modes: ambiguous boundary
cases between categories, and output noise from free-tier models (which
occasionally prepend labels like `User Safety: safe`, breaking JSON parsing).

Adding a resilient parsing step (a `RunnableLambda` that strips known noise
before the JSON parser) eliminated the noise-related failures.

Note: the test set is small (8 cases) and free-tier models are non-deterministic,
so accuracy varies between runs.

## Features

- **Classification** — sorts messages into complaint / order inquiry / product question / other
- **Structured extraction** — pulls name, order number, urgency, and language as JSON
- **Reply drafting** — writes a reply in the customer's own language
- **Smart alerting** — high-urgency messages and complaints trigger an internal team alert, in addition to the customer reply
- **Resilient parsing** — strips output noise from free-tier models before parsing
- **Bilingual UI** — Persian and English text separated into locale files, with RTL support

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