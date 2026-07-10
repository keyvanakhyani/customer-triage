# Customer Message Triage

An AI-powered tool that reads incoming customer messages, classifies them,
extracts structured data, and drafts a reply — in both Persian and English.

Built with LangChain and LCEL, running on free-tier models via OpenRouter.

## The problem

Small businesses receive dozens of customer messages daily across channels.
Each one has to be read, categorized, checked for an order number, and answered.
It's repetitive work that eats hours every week.

This tool automates the first pass: it reads the message, decides what kind of
request it is, pulls out the details that matter, and drafts a reply in the
brand's voice — ready for a human to approve and send.

## Status

🚧 In development. Currently building the core chains.

- [ ] Core: classification chain
- [ ] Core: structured extraction
- [ ] Core: reply drafting
- [ ] i18n: Persian and English locales
- [ ] Gradio web interface
- [ ] Evaluation and accuracy metrics

## Tech stack

- **LangChain** — chain composition with LCEL
- **OpenRouter** — model access (free tier)
- **Gradio** — web interface
- **Python 3.11**