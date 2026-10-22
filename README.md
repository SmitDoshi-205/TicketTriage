# Multi-Agent Customer Support Ticket Triage

A pipeline of five cooperating agents (Classifier, Sentiment, Knowledge, Escalation, Responder)
that classifies support tickets, measures customer sentiment, finds the matching FAQ answer,
decides whether to auto-resolve or escalate to a human, and drafts the reply.
No paid APIs and no LLM calls are used.

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

A full README with results will be added after evaluation.