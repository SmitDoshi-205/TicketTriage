import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import Orchestrator

TICKETS = [
    ("Angry billing",
     "This is the WORST service ever! I was charged twice for my order and I am FURIOUS. Fix this NOW!!!"),
    ("Routine shipping", "How long will delivery take for my order?"),
    ("Vague message", "my thing is not working"),
]


def short(value, limit: int = 90) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    orchestrator = Orchestrator(enable_logging=True)

    for label, text in TICKETS:
        ticket = orchestrator.process(text)
        print("=" * 78)
        print(f"{label.upper()}  (ticket {ticket['ticket_id']})")
        print(f"Text: {text}")
        print("-" * 78)

        for number, step in enumerate(ticket["trace"], start=1):
            print(f"Step {number}: {step['agent']}   [{step['time_ms']} ms]")
            for key, value in step["output"].items():
                if key == "reply":
                    continue                     
                print(f"    {key}: {short(value)}")

        print("-" * 78)
        print(f"FINAL DECISION : {ticket['decision']}   ({ticket['reason']})")
        print(f"TOTAL TIME     : {ticket['total_time_ms']} ms")
        print("REPLY:\n" + ticket["reply"] + "\n")

    print(f"Tickets were appended to: {orchestrator.log_file.relative_to(orchestrator.log_file.parents[1])}")


if __name__ == "__main__":
    main()