import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    print(f"Missing library: {error}. Activate the venv, then: pip install -r requirements.txt")
    sys.exit(1)

import config

# The three standard tickets used throughout the project.
ANGRY = "This is the WORST service ever! I was charged twice for my order and I am FURIOUS. Fix this NOW!!!"
ROUTINE = "How long will delivery take for my order?"
VAGUE = "my thing is not working"

RESULT_FILES = [
    "classifier_test_metrics.csv", "confusion_matrix_test.csv", "confusion_matrix_test.png",
    "classifier_eval_metrics.csv", "confusion_matrix_eval.csv", "confusion_matrix_eval.png",
    "eval_predictions.csv", "escalation_metrics.csv", "timing.csv",
    "baseline_comparison.csv", "ablation.csv",
    "baseline_vs_multiagent.png", "decision_distribution.png", "eval_errors.csv",
]

_CACHE = {}


def get_orchestrator():
    """Create the Orchestrator once and reuse it (logging off)."""
    if "orchestrator" not in _CACHE:
        from orchestrator import Orchestrator
        _CACHE["orchestrator"] = Orchestrator(enable_logging=False)
    return _CACHE["orchestrator"]


class Phase:
    """Collects the checks, failures and key numbers of one phase."""

    def __init__(self) -> None:
        self.total = 0
        self.failures = []
        self.notes = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        """Record one check; only failures are shown in the output."""
        self.total += 1
        if not condition:
            self.failures.append(name + (f"  [{detail}]" if detail else ""))
        return bool(condition)

    def note(self, text: str) -> None:
        """Record a key number to print under the phase."""
        self.notes.append(text)


def run_phase(number: int, title: str, function) -> bool:
    """Run one phase function, catch crashes, print the compact result."""
    phase = Phase()
    try:
        function(phase)
    except Exception as error:                      # a crash counts as a failure
        phase.failures.append(f"crashed: {type(error).__name__}: {error}")

    passed = not phase.failures
    print(f"PHASE {number}  {'PASS' if passed else 'FAIL'}  {title}")
    for note in phase.notes:
        print(f"           {note}")
    if not passed:
        print(f"           {len(phase.failures)} problem(s) in {phase.total} checks:")
        for failure in phase.failures:
            print(f"           - {failure}")
    return passed


# ----------------------------------------------------------------------
# PHASE 1: data
# ----------------------------------------------------------------------
def phase1(p: Phase) -> None:
    """Check the raw copy, train.csv and test.csv."""
    p.check("data/raw/bitext_raw.csv saved", (config.DATA_DIR / "raw" / "bitext_raw.csv").exists())
    if not p.check("train.csv and test.csv exist", config.TRAIN_FILE.exists() and config.TEST_FILE.exists(),
                   "run: python scripts\\prepare_data.py"):
        return

    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    p.check("columns 'text' and 'category' present",
            {"text", "category"} <= set(train.columns) and {"text", "category"} <= set(test.columns))
    p.check("exactly our 5 categories in train and test",
            set(train["category"]) == set(config.CATEGORIES) and set(test["category"]) == set(config.CATEGORIES))

    train_counts = train["category"].value_counts()
    test_counts = test["category"].value_counts()
    p.check("train classes balanced", train_counts.max() - train_counts.min() <= 2, str(train_counts.to_dict()))
    p.check("test classes balanced", test_counts.max() - test_counts.min() <= 2, str(test_counts.to_dict()))

    share = len(test) / (len(train) + len(test))
    p.check("test share is about 20%", 0.19 <= share <= 0.21, f"{share:.3f}")
    p.check("no identical text in train and test", len(set(train["text"]) & set(test["text"])) == 0)
    p.check("no missing text", bool(train["text"].notna().all() and test["text"].notna().all()))

    p.note(f"train {len(train)} rows, test {len(test)} rows ({share * 100:.1f}% test), "
           f"{int(train_counts.min())}-{int(train_counts.max())} per class in train")


# ----------------------------------------------------------------------
# PHASE 2: classifier
# ----------------------------------------------------------------------
def phase2(p: Phase) -> None:
    """Check the saved model and the Classifier Agent."""
    from sklearn.metrics import accuracy_score, f1_score
    from agents.classifier import ClassifierAgent

    if not p.check("models/classifier.joblib exists", config.MODEL_FILE.exists(),
                   "run: python scripts\\train_classifier.py"):
        return
    agent = ClassifierAgent()
    p.check("model knows exactly our 5 categories", sorted(agent.model.classes_) == sorted(config.CATEGORIES))

    test = pd.read_csv(config.TEST_FILE)
    predictions = agent.model.predict(test["text"])
    accuracy = accuracy_score(test["category"], predictions)
    macro_f1 = f1_score(test["category"], predictions, average="macro")
    p.check("test accuracy >= 0.70", accuracy >= 0.70, f"{accuracy:.4f}")

    out = agent.run({"text": "I want a refund for my last payment"})
    p.check("agent output has exactly category, confidence, priority",
            set(out) == {"category", "confidence", "priority"}, str(out))
    p.check("confidence is between 0 and 1", 0.0 <= out["confidence"] <= 1.0)
    p.check("priority matches the category table", out["priority"] == config.PRIORITY_MAP[out["category"]])
    p.check("results/classifier_report.csv saved", (config.RESULTS_DIR / "classifier_report.csv").exists())
    p.check("results/classifier_confusion_matrix.csv saved",
            (config.RESULTS_DIR / "classifier_confusion_matrix.csv").exists())

    p.note(f"test accuracy {accuracy:.4f}, macro F1 {macro_f1:.4f} (templated data, so very high is expected)")


# ----------------------------------------------------------------------
# PHASE 3: sentiment
# ----------------------------------------------------------------------
def phase3(p: Phase) -> None:
    """Check the Sentiment Agent."""
    from agents.sentiment import SentimentAgent

    agent = SentimentAgent()
    calm = agent.run({"text": "Hello, I would like to know when my order will arrive."})
    angry = agent.run({"text": "I am extremely angry. Your service is terrible and I was charged twice!"})
    polite = agent.run({"text": "Thank you so much for your help, I really appreciate it!"})

    p.check("output has exactly sentiment_score and sentiment_label",
            set(calm) == {"sentiment_score", "sentiment_label"}, str(calm))
    p.check("scores are within -1 and +1",
            all(-1 <= r["sentiment_score"] <= 1 for r in (calm, angry, polite)))
    p.check("angry text scores lower than calm text", angry["sentiment_score"] < calm["sentiment_score"])
    p.check("polite text scores positive", polite["sentiment_score"] > config.SENTIMENT_POSITIVE)
    p.check("label rules: Very Negative / Negative / Neutral / Positive",
            [SentimentAgent.label_for(s) for s in (config.SENTIMENT_VERY_NEGATIVE, -0.3, 0.0, 0.5)]
            == ["Very Negative", "Negative", "Neutral", "Positive"])

    p.note(f"calm {calm['sentiment_score']:+.2f}, angry {angry['sentiment_score']:+.2f}, "
           f"polite {polite['sentiment_score']:+.2f}")


# ----------------------------------------------------------------------
# PHASE 4: knowledge
# ----------------------------------------------------------------------
def phase4(p: Phase) -> None:
    """Check faq.csv and the Knowledge Agent (including the fallback search)."""
    from agents.knowledge import KnowledgeAgent

    if not p.check("data/faq.csv exists", config.FAQ_FILE.exists()):
        return
    faq = pd.read_csv(config.FAQ_FILE).fillna("")
    p.check("faq.csv has columns category, question, answer", {"category", "question", "answer"} <= set(faq.columns))
    p.check("faq.csv covers our 5 categories", set(faq["category"]) == set(config.CATEGORIES))
    p.check("faq.csv has at least 20 entries", len(faq) >= 20, f"{len(faq)} rows")
    p.check("no empty answers", bool((faq["answer"].str.strip() != "").all()))
    if not p.check("config.py has FAQ_FALLBACK_THRESHOLD (updated Knowledge Agent)",
                   hasattr(config, "FAQ_FALLBACK_THRESHOLD"), "replace config.py with the latest version"):
        return

    agent = KnowledgeAgent()
    queries = [
        ("I would like a refund for my order, please return my money", "Billing & Refund", "refund"),
        ("My package has not arrived yet, where is it?", "Shipping & Delivery", "not arrived"),
        ("How do I change my password?", "Account & Subscription", "password"),
        ("I need to cancel the order I placed yesterday", "Order & Cancellation", "cancel"),
        ("I want to talk to a real person", "General Inquiry & Feedback", "real person"),
    ]
    matched = 0
    for text, category, keyword in queries:
        result = agent.run({"text": text, "category": category})
        ok = p.check(f"query matches the right FAQ: '{text[:40]}'", keyword in result["faq_question"].lower(),
                     f"got '{result['faq_question']}', similarity {result['faq_similarity']}")
        matched += ok

    out = agent.run({"text": "purple banana helicopter xyz", "category": "General Inquiry & Feedback"})
    p.check("gibberish gets a similarity below the escalation threshold",
            out["faq_similarity"] < config.FAQ_SIMILARITY_THRESHOLD, str(out["faq_similarity"]))
    p.check("output has the 4 knowledge fields",
            set(out) == {"faq_question", "faq_answer", "faq_similarity", "faq_scope"}, str(sorted(out)))

    unknown = agent.run({"text": "How do I reset my password?", "category": "Unknown"})
    p.check("unknown category searches all FAQs", unknown["faq_scope"] == "all" and "password" in unknown["faq_question"].lower())

    # A classifier mistake must not hide the right answer (fallback search).
    wrong = agent.run({"text": "How can I track my package?", "category": "Billing & Refund"})
    p.check("wrong category: fallback still finds the tracking FAQ",
            wrong["faq_scope"] == "all_fallback" and "track" in wrong["faq_question"].lower(),
            f"scope {wrong['faq_scope']}, matched '{wrong['faq_question']}'")

    p.note(f"{len(faq)} FAQs, {matched}/{len(queries)} test queries matched, "
           f"gibberish similarity {out['faq_similarity']}, wrong-category fallback tested")


# ----------------------------------------------------------------------
# PHASE 5: escalation + responder
# ----------------------------------------------------------------------
def phase5(p: Phase) -> None:
    """Check every escalation rule, rule order, boundaries and the 3 reply templates."""
    from agents.escalation import EscalationAgent
    from agents.responder import ResponderAgent

    escalation, responder = EscalationAgent(), ResponderAgent()

    def base(drop=(), **overrides) -> dict:
        """A healthy routine ticket; override or drop fields per case."""
        ticket = {
            "text": ROUTINE, "category": "Shipping & Delivery", "confidence": 0.90,
            "priority": "Medium", "sentiment_score": 0.0, "sentiment_label": "Neutral",
            "faq_question": "How long does delivery take?",
            "faq_answer": "Standard delivery takes 3 to 7 business days depending on your location.",
            "faq_similarity": 0.60,
        }
        ticket.update(overrides)
        for key in drop:
            ticket.pop(key, None)
        return ticket

    # (name, ticket, expected decision, expected reason_code, expected reply_type or None)
    cases = [
        ("angry billing -> ESCALATE", base(text=ANGRY, category="Billing & Refund", priority="High",
            confidence=0.97, sentiment_score=-0.82, sentiment_label="Very Negative", faq_similarity=0.55),
            "ESCALATE", "VERY_ANGRY", "ESCALATION"),
        ("routine shipping -> AUTO_RESOLVE", base(), "AUTO_RESOLVE", "ROUTINE", "AUTO_RESOLVE"),
        ("vague message -> ESCALATE", base(text=VAGUE, category="General Inquiry & Feedback", priority="Low",
            confidence=0.31, sentiment_score=-0.20, sentiment_label="Negative", faq_similarity=0.05),
            "ESCALATE", "LOW_CONFIDENCE", "MORE_DETAILS"),
        ("legal keyword", base(text="I will take legal action against you"), "ESCALATE", "LEGAL_FRAUD", "ESCALATION"),
        ("'issue' does not match keyword 'sue'", base(text="I have an issue with my order"), "AUTO_RESOLVE", "ROUTINE", None),
        ("keyword beats low confidence (rule order)", base(text="my lawyer will call you", confidence=0.20),
            "ESCALATE", "LEGAL_FRAUD", None),
        ("confidence exactly at threshold is not low", base(confidence=config.CONFIDENCE_THRESHOLD),
            "AUTO_RESOLVE", "ROUTINE", None),
        ("FAQ similarity just below threshold", base(faq_similarity=config.FAQ_SIMILARITY_THRESHOLD - 0.01),
            "ESCALATE", "LOW_FAQ_SIMILARITY", "ESCALATION"),
        ("sentiment exactly at very-angry threshold", base(sentiment_score=config.SENTIMENT_VERY_ANGRY),
            "ESCALATE", "VERY_ANGRY", None),
        ("angry threshold + High priority", base(sentiment_score=config.SENTIMENT_ANGRY, priority="High"),
            "ESCALATE", "ANGRY_HIGH_PRIORITY", None),
        ("angry threshold + Medium priority does not escalate",
            base(sentiment_score=config.SENTIMENT_ANGRY, priority="Medium"), "AUTO_RESOLVE", "ROUTINE", None),
        ("missing sentiment fields: rules skipped", base(drop=("sentiment_score", "sentiment_label")),
            "AUTO_RESOLVE", "ROUTINE", None),
    ]

    replies = {}
    passed = 0
    for name, ticket, want_decision, want_code, want_type in cases:
        ticket.update(escalation.run(ticket))
        ticket.update(responder.run(ticket))
        replies[name] = ticket["reply"]
        ok = (ticket["decision"] == want_decision and ticket["reason_code"] == want_code
              and (want_type is None or ticket["reply_type"] == want_type))
        passed += p.check(name, ok, f"got {ticket['decision']} / {ticket['reason_code']} / {ticket['reply_type']}")

    p.check("escalation reply promises a human agent", "human agent" in replies["angry billing -> ESCALATE"])
    p.check("auto reply contains the FAQ answer", "3 to 7 business days" in replies["routine shipping -> AUTO_RESOLVE"])
    p.check("'more details' reply asks for details", "more details" in replies["vague message -> ESCALATE"])

    p.note(f"{passed}/{len(cases)} rule and boundary cases correct, 3 reply templates verified")


# ----------------------------------------------------------------------
# PHASE 6: orchestrator
# ----------------------------------------------------------------------
def phase6(p: Phase) -> None:
    """Check the real pipeline on the 3 standard tickets, the trace, timing and the log CSV."""
    from orchestrator import Orchestrator

    p.check("scripts/run_demo.py exists", (ROOT / "scripts" / "run_demo.py").exists())
    p.check("tests/smoke_test.py exists", (ROOT / "tests" / "smoke_test.py").exists())

    orchestrator = get_orchestrator()
    expected_order = ["Classifier Agent", "Sentiment Agent", "Knowledge Agent", "Escalation Agent", "Responder Agent"]
    required = {"ticket_id", "text", "category", "confidence", "priority", "sentiment_score", "sentiment_label",
                "faq_question", "faq_answer", "faq_similarity", "decision", "reason", "reason_code",
                "reply", "reply_type", "trace", "timings_ms", "total_time_ms"}

    summary, times = [], []
    for label, text, expected in [("angry billing", ANGRY, "ESCALATE"),
                                  ("routine shipping", ROUTINE, "AUTO_RESOLVE"),
                                  ("vague message", VAGUE, "ESCALATE")]:
        ticket = orchestrator.process(text)
        p.check(f"{label} -> {expected}", ticket["decision"] == expected,
                f"got {ticket['decision']}: {ticket['reason']}")
        p.check(f"{label}: all fields present", required <= set(ticket), f"missing {sorted(required - set(ticket))}")
        p.check(f"{label}: trace has the 5 agents in order", [s["agent"] for s in ticket["trace"]] == expected_order)
        p.check(f"{label}: every agent timed", len(ticket["timings_ms"]) == 5 and ticket["total_time_ms"] > 0)
        summary.append(f"{label}: {ticket['decision']} ({ticket['reason_code']})")
        times.append(ticket["total_time_ms"])

    many = orchestrator.process_many([ROUTINE, VAGUE])
    p.check("process_many returns one ticket per text", len(many) == 2)

    # Logging is tested on a temporary file, never on results/ticket_log.csv.
    with tempfile.TemporaryDirectory() as folder:
        log_path = Path(folder) / "log.csv"
        logged = Orchestrator(enable_logging=True, log_file=log_path)
        logged.process(ROUTINE)
        logged.process(VAGUE)
        rows = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        p.check("log CSV gets one row per ticket", len(rows) == 2, f"{len(rows)} rows")
        p.check("log CSV has decision, reason and reply columns", {"decision", "reason", "reply"} <= set(rows.columns))

    p.note(" | ".join(summary))
    p.note(f"pipeline time per ticket: {np.mean(times):.1f} ms on average (3 tickets)")


# ----------------------------------------------------------------------
# PHASE 7: evaluation
# ----------------------------------------------------------------------
def phase7(p: Phase, run_eval: bool) -> None:
    """Check eval_set.csv, the saved results, and that those results match the current code."""
    if not p.check("data/eval_set.csv exists", config.EVAL_FILE.exists()):
        return
    eval_df = pd.read_csv(config.EVAL_FILE)
    if not p.check("eval_set.csv has columns text, true_category, should_escalate",
                   {"text", "true_category", "should_escalate"} <= set(eval_df.columns)):
        return
    eval_df["should_escalate"] = eval_df["should_escalate"].astype(str).str.strip().str.lower()
    valid = (set(eval_df["true_category"]) <= set(config.CATEGORIES)
             and set(eval_df["should_escalate"]) <= {"yes", "no"})
    if not p.check("eval_set.csv: valid category names and yes/no flags", valid):
        return
    should = (eval_df["should_escalate"] == "yes").to_numpy()
    p.check("at least 30 eval tickets", len(eval_df) >= 30, f"{len(eval_df)}")
    p.check("roughly 30-50% of eval tickets should escalate", 0.30 <= should.mean() <= 0.50,
            f"{should.mean() * 100:.0f}%")

    if run_eval:
        outcome = subprocess.run([sys.executable, str(ROOT / "evaluate.py")], cwd=ROOT,
                                 capture_output=True, text=True)
        p.check("evaluate.py ran without error", outcome.returncode == 0, outcome.stderr.strip()[-300:])

    missing = [name for name in RESULT_FILES if not (config.RESULTS_DIR / name).exists()]
    if not p.check("all evaluate.py result files exist in results/", not missing,
                   f"missing {missing}; run: python evaluate.py"):
        return
    p.check("chart PNGs are not empty",
            all((config.RESULTS_DIR / n).stat().st_size > 1000 for n in RESULT_FILES if n.endswith(".png")))

    # Re-run the current code on the eval set and compare with the saved results.
    orchestrator = get_orchestrator()
    tickets = [orchestrator.process(text, log=False) for text in eval_df["text"]]
    decisions = [t["decision"] for t in tickets]
    predicted = np.array([d == "ESCALATE" for d in decisions])

    saved = pd.read_csv(config.RESULTS_DIR / "eval_predictions.csv")
    same_size = len(saved) == len(eval_df)
    p.check("eval_predictions.csv has one row per eval ticket", same_size, f"{len(saved)} vs {len(eval_df)}")
    if same_size:
        different = sum(a != b for a, b in zip(saved["decision"], decisions))
        p.check("saved results match the current code (not stale)", different == 0,
                f"{different} decisions differ: run python evaluate.py again")

    tp = int(np.sum(should & predicted))
    fp = int(np.sum(~should & predicted))
    fn = int(np.sum(should & ~predicted))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auto_rate = float(np.mean(~predicted))

    metrics = pd.read_csv(config.RESULTS_DIR / "escalation_metrics.csv").iloc[0]
    p.check("escalation_metrics.csv matches recomputed recall and precision",
            abs(metrics["escalation_recall"] - recall) < 0.002 and abs(metrics["escalation_precision"] - precision) < 0.002,
            "stale file: run python evaluate.py again")

    comparison = pd.read_csv(config.RESULTS_DIR / "baseline_comparison.csv")
    baseline = comparison.iloc[0]
    p.check("baseline_comparison.csv has Baseline A and the full system",
            len(comparison) == 2 and "Baseline A" in str(baseline["system"]))
    p.check("Baseline A never escalates (recall 0, auto-resolve rate 1.0)",
            baseline["escalation_recall"] == 0 and baseline["auto_resolve_rate"] == 1.0)
    p.check("full system escalation recall beats Baseline A", recall > baseline["escalation_recall"], f"recall {recall:.3f}")
    p.check("ablation.csv has 4 rows (full + 3 agents removed)", len(pd.read_csv(config.RESULTS_DIR / "ablation.csv")) == 4)

    accuracy_eval = float(np.mean([t["category"] == c for t, c in zip(tickets, eval_df["true_category"])]))
    test_metrics = pd.read_csv(config.RESULTS_DIR / "classifier_test_metrics.csv").set_index("metric")["value"]
    timing = pd.read_csv(config.RESULTS_DIR / "timing.csv").set_index("agent")["mean_ms"]
    reasons = Counter(t["reason_code"] for t in tickets)

    p.note(f"eval set: {len(eval_df)} tickets, {int(should.sum())} should escalate, system escalated {int(predicted.sum())}")
    p.note(f"escalation: recall {recall:.3f}, precision {precision:.3f}, F1 {f1:.3f}, auto-resolve rate {auto_rate:.3f} "
           f"(missed {fn}, false escalations {fp})")
    p.note(f"Baseline A recall {baseline['escalation_recall']:.3f} -> full system recall {recall:.3f}")
    p.note(f"classifier accuracy: test split {test_metrics['accuracy']:.3f} vs messy eval set {accuracy_eval:.3f}")
    p.note("decision reasons: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    p.note(f"average time per ticket: {timing.get('WHOLE PIPELINE', float('nan')):.1f} ms")


def main() -> None:
    """Run all seven phases and print the verdict."""
    run_eval = "--run-eval" in sys.argv
    print(f"Python {sys.version.split()[0]} | combined check of Phases 1-7")
    print("-" * 64)

    phases = [
        (1, "Data preparation", phase1),
        (2, "Classifier Agent", phase2),
        (3, "Sentiment Agent", phase3),
        (4, "Knowledge Agent", phase4),
        (5, "Escalation + Responder Agents", phase5),
        (6, "Orchestrator", phase6),
        (7, "Evaluation", lambda p: phase7(p, run_eval)),
    ]
    results = [run_phase(number, title, function) for number, title, function in phases]

    print("-" * 64)
    print(f"RESULT: {sum(results)}/{len(results)} phases pass")
    if all(results):
        print("ALL 7 PHASES PASS")
    else:
        print("Paste this whole output in the chat.")
        sys.exit(1)


if __name__ == "__main__":
    main()