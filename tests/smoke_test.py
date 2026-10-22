import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import Orchestrator

EXPECTED = [
    ("Angry billing",
     "This is the WORST service ever! I was charged twice for my order and I am FURIOUS. Fix this NOW!!!",
     "ESCALATE"),
    ("Routine shipping", "How long will delivery take for my order?", "AUTO_RESOLVE"),
    ("Vague message", "my thing is not working", "ESCALATE"),
]
EXPECTED_AGENT_ORDER = [
    "Classifier Agent", "Sentiment Agent", "Knowledge Agent",
    "Escalation Agent", "Responder Agent",
]

failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    if not condition:
        failures += 1
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    orchestrator = Orchestrator(enable_logging=False)

    for label, text, expected in EXPECTED:
        ticket = orchestrator.process(text)
        check(f"{label} -> {expected}", ticket["decision"] == expected,
              f"got {ticket['decision']}: {ticket['reason']}")

    # Structure checks on the last ticket.
    check("trace lists all 5 agents in order",
          [step["agent"] for step in ticket["trace"]] == EXPECTED_AGENT_ORDER)
    check("every agent has a timing", len(ticket["timings_ms"]) == 5)
    check("total time recorded", ticket["total_time_ms"] > 0, f"{ticket['total_time_ms']} ms")
    check("reply was drafted", len(ticket["reply"]) > 20)

    print()
    if failures == 0:
        print("SMOKE TEST: PASS")
    else:
        print(f"SMOKE TEST: FAIL ({failures} problem(s)). Paste the output in chat.")
        sys.exit(1)


if __name__ == "__main__":
    main()