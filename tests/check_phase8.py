import py_compile
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

ANGRY = "This is the WORST service ever! I was charged twice for my order and I am FURIOUS. Fix this NOW!!!"
ROUTINE = "How long will delivery take for my order?"
VAGUE = "my thing is not working"

failures = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global failures
    if not condition:
        failures += 1
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def part_a_responder() -> None:
    from agents.responder import ResponderAgent
    from orchestrator import Orchestrator

    print("--- A. Responder fix")
    responder = ResponderAgent()

    def reply_for(score: float, label: str) -> dict:
        return responder.run({
            "decision": "ESCALATE", "reason_code": "LOW_CONFIDENCE",
            "category": "General Inquiry & Feedback", "sentiment_score": score,
            "sentiment_label": label, "faq_question": "", "faq_answer": "", "faq_similarity": 0.0,
        })

    neutral = reply_for(0.0, "Neutral")
    mild = reply_for(-0.20, "Negative")
    furious = reply_for(-0.92, "Very Negative")
    check("unclear + neutral customer -> MORE_DETAILS reply", neutral["reply_type"] == "MORE_DETAILS")
    check("unclear + mildly negative (-0.20) -> MORE_DETAILS reply", mild["reply_type"] == "MORE_DETAILS")
    check("unclear + furious customer -> ESCALATION reply",
          furious["reply_type"] == "ESCALATION" and "human agent" in furious["reply"]
          and "more details" not in furious["reply"], furious["reply_type"])

    orchestrator = Orchestrator(enable_logging=False)
    angry = orchestrator.process(ANGRY)
    check("real pipeline: angry billing ticket gets the ESCALATION reply",
          angry["decision"] == "ESCALATE" and angry["reply_type"] == "ESCALATION",
          f"{angry['decision']} / {angry['reply_type']} / {angry['reason_code']}")
    vague = orchestrator.process(VAGUE)
    check("real pipeline: vague ticket gets the MORE_DETAILS reply",
          vague["decision"] == "ESCALATE" and vague["reply_type"] == "MORE_DETAILS",
          f"{vague['decision']} / {vague['reply_type']} / {vague['reason_code']}")
    routine = orchestrator.process(ROUTINE)
    check("real pipeline: routine ticket gets the AUTO_RESOLVE reply",
          routine["decision"] == "AUTO_RESOLVE" and routine["reply_type"] == "AUTO_RESOLVE",
          f"{routine['decision']} / {routine['reason']}")


def part_b_compile() -> bool:
    print("--- B. app.py")
    app_path = ROOT / "app.py"
    if not check("app.py exists in the project root", app_path.exists()):
        return False
    try:
        py_compile.compile(str(app_path), doraise=True)
        return check("app.py compiles", True)
    except py_compile.PyCompileError as error:
        return check("app.py compiles", False, str(error)[:300])


def part_c_app() -> None:
    print("--- C. Headless run of the app")
    try:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        errors = [str(e.value) for e in at.exception]
        if not check("app starts without an exception", not errors, "; ".join(errors)[:400]):
            return

        examples = [b for b in at.button if b.key and b.key.startswith("example_")]
        check("5 example buttons exist", len(examples) == 5, f"found {len(examples)}")

        at.checkbox(key="save_to_log").set_value(False).run()      # keep the real log clean
        check("log checkbox switched off for this test", at.session_state["save_to_log"] is False)

        standard = [(ANGRY, "ESCALATE"), (ROUTINE, "AUTO_RESOLVE"), (VAGUE, "ESCALATE")]
        for index, (text, expected) in enumerate(standard):
            at.button(key=f"example_{index}").click().run()
            check(f"example button {index + 1} fills the text box", at.session_state["ticket_text"] == text)

            at.button(key="analyse_button").click().run()
            result = at.session_state["result"]
            check(f"example {index + 1} -> {expected}", result["decision"] == expected,
                  f"got {result['decision']}: {result['reason']}")
            check(f"example {index + 1}: decision badge is shown",
                  any(result["decision"] in markdown.value for markdown in at.markdown))
            check(f"example {index + 1}: trace shows 5 agent steps", len(at.expander) == 5,
                  f"found {len(at.expander)}")
    except Exception as error:
        check("headless app test could run", False, f"{type(error).__name__}: {error}"[:400])


def main() -> None:
    part_a_responder()
    if part_b_compile():
        part_c_app()

    print()
    if failures == 0:
        print("PHASE 8 CHECKPOINT: PASS (also do the manual browser check below)")
    else:
        print(f"PHASE 8 CHECKPOINT: FAIL ({failures} problem(s)). Paste this output in the chat.")
        sys.exit(1)


if __name__ == "__main__":
    main()