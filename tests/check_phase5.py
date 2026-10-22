import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.escalation import EscalationAgent
from agents.responder import ResponderAgent

escalation = EscalationAgent()
responder = ResponderAgent()
results = []


def base_ticket(drop=(), **overrides) -> dict:
    ticket = {
        "text": "How long will delivery take for my order?",
        "category": "Shipping & Delivery",
        "confidence": 0.90,
        "priority": "Medium",
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "faq_question": "How long does delivery take?",
        "faq_answer": "Standard delivery takes 3 to 7 business days depending on your location.",
        "faq_similarity": 0.60,
    }
    ticket.update(overrides)
    for key in drop:
        ticket.pop(key, None)
    return ticket


def run_agents(ticket: dict) -> dict:
    ticket.update(escalation.run(ticket))
    ticket.update(responder.run(ticket))
    return ticket


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    print("=== THREE STANDARD TICKETS ===")
    standard = [
        ("Angry billing", base_ticket(
            text="I was charged twice and nobody is helping me! THIS IS UNACCEPTABLE!!!",
            category="Billing & Refund", priority="High", confidence=0.97,
            sentiment_score=-0.82, sentiment_label="Very Negative",
            faq_question="I was charged twice for the same order. What should I do?",
            faq_answer="Duplicate charges are usually temporary holds that disappear in 3 to 5 days.",
            faq_similarity=0.55), "ESCALATE", "VERY_ANGRY", "ESCALATION"),
        ("Routine shipping", base_ticket(), "AUTO_RESOLVE", "ROUTINE", "AUTO_RESOLVE"),
        ("Vague message", base_ticket(
            text="my thing is not working", category="General Inquiry & Feedback",
            priority="Low", confidence=0.31, sentiment_score=-0.20,
            sentiment_label="Negative", faq_similarity=0.05), "ESCALATE", "LOW_CONFIDENCE", "MORE_DETAILS"),
    ]
    replies = {}
    for label, ticket, want_decision, want_code, want_type in standard:
        run_agents(ticket)
        replies[label] = ticket["reply"]
        check(f"{label} -> {want_decision}",
              ticket["decision"] == want_decision and ticket["reason_code"] == want_code
              and ticket["reply_type"] == want_type,
              f"{ticket['decision']} | {ticket['reason']} | reply: {ticket['reply_type']}")

    print("\n=== EDGE CASES (rule order, boundaries, missing fields) ===")
    edge_cases = [
        ("legal keyword", base_ticket(text="I will take legal action against you"), "ESCALATE", "LEGAL_FRAUD"),
        ("'issue' must NOT match keyword 'sue'", base_ticket(text="I have an issue with my order"), "AUTO_RESOLVE", "ROUTINE"),
        ("keyword beats low confidence (rule order)", base_ticket(text="my lawyer will call you", confidence=0.20), "ESCALATE", "LEGAL_FRAUD"),
        ("confidence exactly 0.50 is NOT low", base_ticket(confidence=0.50), "AUTO_RESOLVE", "ROUTINE"),
        ("FAQ similarity 0.29 is low", base_ticket(faq_similarity=0.29), "ESCALATE", "LOW_FAQ_SIMILARITY"),
        ("sentiment exactly -0.60 escalates", base_ticket(sentiment_score=-0.60), "ESCALATE", "VERY_ANGRY"),
        ("-0.30 with High priority escalates", base_ticket(sentiment_score=-0.30, priority="High"), "ESCALATE", "ANGRY_HIGH_PRIORITY"),
        ("-0.30 with Medium priority does NOT", base_ticket(sentiment_score=-0.30, priority="Medium"), "AUTO_RESOLVE", "ROUTINE"),
        ("missing sentiment fields -> rules skipped", base_ticket(drop=("sentiment_score", "sentiment_label")), "AUTO_RESOLVE", "ROUTINE"),
    ]
    for name, ticket, want_decision, want_code in edge_cases:
        run_agents(ticket)
        check(name, ticket["decision"] == want_decision and ticket["reason_code"] == want_code,
              f"{ticket['decision']} / {ticket['reason_code']}")

    print("\n=== REPLY CONTENT ===")
    check("escalation reply promises a human agent", "human agent" in replies["Angry billing"])
    check("auto reply contains the FAQ answer", "3 to 7 business days" in replies["Routine shipping"])
    check("vague reply asks for more details", "more details" in replies["Vague message"])

    print("\n--- Sample reply: ANGRY BILLING ---\n" + replies["Angry billing"])
    print("\n--- Sample reply: VAGUE MESSAGE ---\n" + replies["Vague message"])

    passed = sum(results)
    print(f"\n=== {passed}/{len(results)} checks passed ===")
    if passed == len(results):
        print("PHASE 5 CHECKPOINT: PASS")
    else:
        print("PHASE 5 CHECKPOINT: FAIL (paste the output in chat)")
        sys.exit(1)


if __name__ == "__main__":
    main()