import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config
from orchestrator import Orchestrator

RESULTS = config.RESULTS_DIR
SHORT = {name: name.split()[0] for name in config.CATEGORIES}   # Billing, Order, ...


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_eval_set() -> pd.DataFrame:
    if not config.EVAL_FILE.exists():
        print("STOP: data/eval_set.csv is missing. Create it from the chat.")
        sys.exit(1)
    df = pd.read_csv(config.EVAL_FILE)
    if not {"text", "true_category", "should_escalate"}.issubset(df.columns):
        print("STOP: eval_set.csv must have columns: text, true_category, should_escalate")
        print("Found:", list(df.columns))
        sys.exit(1)

    df["should_escalate"] = df["should_escalate"].astype(str).str.strip().str.lower()
    bad_categories = sorted(set(df["true_category"]) - set(config.CATEGORIES))
    bad_flags = sorted(set(df["should_escalate"]) - {"yes", "no"})
    if bad_categories or bad_flags:
        print("STOP: invalid values in eval_set.csv")
        print("  unknown categories:", bad_categories)
        print("  should_escalate values other than yes/no:", bad_flags)
        sys.exit(1)
    return df


def save_confusion_png(y_true, y_pred, title: str, path: Path) -> pd.DataFrame:
    labels = config.CATEGORIES
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    names = [SHORT[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return pd.DataFrame(matrix, index=[f"true_{n}" for n in names],
                        columns=[f"pred_{n}" for n in names])


def escalation_metrics(name: str, should: np.ndarray, predicted: np.ndarray) -> dict:
   
    tp = int(np.sum(should & predicted))       # correctly escalated
    fp = int(np.sum(~should & predicted))      # escalated but routine
    fn = int(np.sum(should & ~predicted))      # needed a human but auto-replied
    precision = tp / (tp + fp) if (tp + fp) else 0.0     # 0 when nothing was escalated
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "system": name,
        "escalation_precision": round(precision, 3),
        "escalation_recall": round(recall, 3),
        "escalation_f1": round(f1, 3),
        "auto_resolve_rate": round(float(np.mean(~predicted)), 3),
        "missed_escalations": fn,
        "false_escalations": fp,
    }


# ----------------------------------------------------------------------
# (a) classifier on the held-out Bitext test split
# ----------------------------------------------------------------------
def evaluate_test_split(orch: Orchestrator) -> None:
    """Accuracy, macro F1 and confusion matrix on data/test.csv."""
    test_df = pd.read_csv(config.TEST_FILE)
    predictions = orch.classifier.model.predict(test_df["text"])

    accuracy = accuracy_score(test_df["category"], predictions)
    macro_f1 = f1_score(test_df["category"], predictions, average="macro")
    pd.DataFrame([
        {"metric": "accuracy", "value": round(accuracy, 4)},
        {"metric": "macro_f1", "value": round(macro_f1, 4)},
        {"metric": "n_test_tickets", "value": len(test_df)},
    ]).to_csv(RESULTS / "classifier_test_metrics.csv", index=False)

    matrix_df = save_confusion_png(test_df["category"], predictions,
                                   "Classifier on Bitext test split",
                                   RESULTS / "confusion_matrix_test.png")
    matrix_df.to_csv(RESULTS / "confusion_matrix_test.csv")

    print("\n=== (a) CLASSIFIER ON BITEXT TEST SPLIT ===")
    print(f"Tickets: {len(test_df)} | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")
    print(matrix_df.to_string())


# ----------------------------------------------------------------------
# (b)-(e) full system on the messy eval set
# ----------------------------------------------------------------------
def run_full_system(orch: Orchestrator, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Run every eval ticket through the real Orchestrator (logging off)."""
    orch.process("warm up the models", log=False)       # first call is slower; do not time it

    rows = []
    for _, row in eval_df.iterrows():
        ticket = orch.process(row["text"], log=False)
        record = {
            "text": row["text"],
            "true_category": row["true_category"],
            "pred_category": ticket["category"],
            "confidence": ticket["confidence"],
            "sentiment_score": ticket["sentiment_score"],
            "faq_similarity": ticket["faq_similarity"],
            "should_escalate": row["should_escalate"],
            "decision": ticket["decision"],
            "reason_code": ticket["reason_code"],
            "reason": ticket["reason"],
            "total_time_ms": ticket["total_time_ms"],
        }
        for agent_name, ms in ticket["timings_ms"].items():
            record[f"time_{agent_name}"] = ms
        rows.append(record)

    results = pd.DataFrame(rows)
    results["category_correct"] = results["true_category"] == results["pred_category"]
    results["escalated"] = results["decision"] == "ESCALATE"
    results["should_flag"] = results["should_escalate"] == "yes"
    results["decision_correct"] = results["escalated"] == results["should_flag"]
    return results


def run_variant(orch: Orchestrator, eval_df: pd.DataFrame, removed: str = None) -> np.ndarray:
    
    escalated = []
    for text in eval_df["text"]:
        ticket = {"text": text}
        if removed != "Classifier":
            ticket.update(orch.classifier.run(ticket))
        if removed != "Sentiment":
            ticket.update(orch.sentiment.run(ticket))
        if removed != "Knowledge":
            ticket.update(orch.knowledge.run(ticket))
        ticket.update(orch.escalation.run(ticket))
        escalated.append(ticket["decision"] == "ESCALATE")
    return np.array(escalated)


def evaluate_eval_set(orch: Orchestrator) -> None:
    eval_df = load_eval_set()
    results = run_full_system(orch, eval_df)
    results.to_csv(RESULTS / "eval_predictions.csv", index=False)
    n = len(results)

    # ---- (b) classifier on messy tickets ----
    accuracy = results["category_correct"].mean()
    macro_f1 = f1_score(results["true_category"], results["pred_category"],
                        labels=config.CATEGORIES, average="macro", zero_division=0)
    pd.DataFrame([
        {"metric": "accuracy", "value": round(float(accuracy), 4)},
        {"metric": "macro_f1", "value": round(float(macro_f1), 4)},
        {"metric": "n_eval_tickets", "value": n},
    ]).to_csv(RESULTS / "classifier_eval_metrics.csv", index=False)
    matrix_df = save_confusion_png(results["true_category"], results["pred_category"],
                                   "Classifier on messy eval set",
                                   RESULTS / "confusion_matrix_eval.png")
    matrix_df.to_csv(RESULTS / "confusion_matrix_eval.csv")

    print("\n=== (b) CLASSIFIER ON MESSY EVAL SET ===")
    print(f"Tickets: {n} | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

    # ---- (c) escalation metrics + (d) auto-resolution rate ----
    should = results["should_flag"].to_numpy()
    full_pred = results["escalated"].to_numpy()
    full = escalation_metrics("Full multi-agent system", should, full_pred)
    pd.DataFrame([full]).to_csv(RESULTS / "escalation_metrics.csv", index=False)

    print("\n=== (c)+(d) ESCALATION METRICS, FULL SYSTEM ===")
    print(f"Should escalate: {int(should.sum())} of {n} | System escalated: {int(full_pred.sum())}")
    print(pd.DataFrame([full]).to_string(index=False))

    # ---- (e) timing ----
    time_columns = [c for c in results.columns if c.startswith("time_")]
    timing = pd.DataFrame({
        "agent": [c.replace("time_", "") for c in time_columns] + ["WHOLE PIPELINE"],
        "mean_ms": [round(results[c].mean(), 2) for c in time_columns]
                   + [round(results["total_time_ms"].mean(), 2)],
    })
    timing.to_csv(RESULTS / "timing.csv", index=False)
    print("\n=== (e) AVERAGE PROCESSING TIME PER TICKET ===")
    print(timing.to_string(index=False))

    # ---- Baseline A: classifier + top FAQ, auto-resolves everything ----
    baseline_pred = np.zeros(n, dtype=bool)                     # never escalates
    baseline = escalation_metrics("Baseline A (classifier only)", should, baseline_pred)
    wrong_cat = ~results["category_correct"].to_numpy()
    baseline["auto_replies_wrong_category"] = int(np.sum(~baseline_pred & wrong_cat))
    full["auto_replies_wrong_category"] = int(np.sum(~full_pred & wrong_cat))
    comparison = pd.DataFrame([baseline, full])
    comparison.to_csv(RESULTS / "baseline_comparison.csv", index=False)

    print("\n=== BASELINE A vs FULL MULTI-AGENT SYSTEM ===")
    print(comparison.to_string(index=False))
    print("(Baseline A never escalates, so its recall is 0 by construction.)")

    # ---- Ablation: remove one agent at a time ----
    ablation_rows = []
    for label, removed in [("Full system", None), ("Without Classifier Agent", "Classifier"),
                           ("Without Sentiment Agent", "Sentiment"),
                           ("Without Knowledge Agent", "Knowledge")]:
        predicted = run_variant(orch, eval_df, removed)
        if removed is None and not np.array_equal(predicted, full_pred):
            print("WARNING: manual full run differs from the Orchestrator run. Tell me.")
        ablation_rows.append(escalation_metrics(label, should, predicted))
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(RESULTS / "ablation.csv", index=False)
    print("\n=== ABLATION (one agent removed at a time) ===")
    print(ablation.to_string(index=False))

    # ---- Chart: baseline vs multi-agent ----
    metrics = ["escalation_recall", "escalation_precision", "escalation_f1"]
    labels = ["Recall", "Precision", "F1"]
    positions = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(positions - 0.2, [baseline[m] for m in metrics], 0.4,
           label="Baseline A (classifier only)", color="#9aa5b1")
    ax.bar(positions + 0.2, [full[m] for m in metrics], 0.4,
           label="Full multi-agent system", color="#2f80ed")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (escalation = positive class)")
    ax.set_title("Escalation quality on the messy eval set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_vs_multiagent.png", dpi=150)
    plt.close(fig)

    # ---- Chart: decision distribution ----
    counts = results["reason_code"].value_counts()
    colours = ["#27ae60" if code == "ROUTINE" else "#eb5757" for code in counts.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=colours)
    ax.set_ylabel("Number of eval tickets")
    ax.set_title("Decisions by reason (green = auto-resolved, red = escalated)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS / "decision_distribution.png", dpi=150)
    plt.close(fig)

    # ---- Wrong decisions, for the error analysis in the report ----
    errors = results[~results["decision_correct"]][
        ["text", "should_escalate", "decision", "reason", "true_category", "pred_category",
         "confidence", "sentiment_score", "faq_similarity"]]
    errors.to_csv(RESULTS / "eval_errors.csv", index=False)
    print(f"\n=== WRONG DECISIONS: {len(errors)} of {n} ===")
    for _, row in errors.iterrows():
        print(f"- [{row['should_escalate']} expected, got {row['decision']}] {row['text']}")
        print(f"    reason: {row['reason']}")


def main() -> None:
    """Run the whole evaluation."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(enable_logging=False)
    evaluate_test_split(orch)
    evaluate_eval_set(orch)

    print("\nSaved files in results/:")
    for path in sorted(RESULTS.iterdir()):
        print("  ", path.name)


if __name__ == "__main__":
    main()