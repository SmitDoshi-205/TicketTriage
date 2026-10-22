"""
diagnose_scores.py
------------------
Diagnostic (not part of the final system). For every ticket in
data/eval_set.csv it prints the classifier confidence, the sentiment score
and FOUR versions of the FAQ similarity:

  sim_cat_QA  : search predicted category only, FAQ = question + answer (current design)
  sim_cat_Q   : search predicted category only, FAQ = question only
  sim_all_QA  : search ALL FAQs, FAQ = question + answer
  sim_all_Q   : search ALL FAQs, FAQ = question only

It then shows how many routine tickets (should_escalate = no) each
escalation threshold would wrongly escalate, and what the three standard
tickets from the smoke test score.

Run: python scripts\\diagnose_scores.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from agents.classifier import ClassifierAgent
from agents.sentiment import SentimentAgent

STANDARD_TICKETS = [
    "This is the WORST service ever! I was charged twice for my order and I am FURIOUS. Fix this NOW!!!",
    "How long will delivery take for my order?",
    "my thing is not working",
]
VARIANTS = ["sim_cat_QA", "sim_cat_Q", "sim_all_QA", "sim_all_Q"]


def build_index(documents):
    """Fit a TF-IDF vectorizer on the documents and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    return vectorizer, vectorizer.fit_transform(documents)


def best_similarity(index, text, rows) -> float:
    """Highest cosine similarity between the text and the chosen FAQ rows."""
    vectorizer, matrix = index
    return float(cosine_similarity(vectorizer.transform([text]), matrix[rows])[0].max())


def score_ticket(text, classifier, sentiment, faq, indexes) -> dict:
    """Run classifier + sentiment and compute the four similarity versions."""
    prediction = classifier.run({"text": text})
    rows_cat = faq.index[faq["category"] == prediction["category"]].tolist()
    rows_all = faq.index.tolist()
    return {
        "pred_category": prediction["category"],
        "confidence": prediction["confidence"],
        "sentiment": sentiment.run({"text": text})["sentiment_score"],
        "sim_cat_QA": best_similarity(indexes["QA"], text, rows_cat),
        "sim_cat_Q": best_similarity(indexes["Q"], text, rows_cat),
        "sim_all_QA": best_similarity(indexes["QA"], text, rows_all),
        "sim_all_Q": best_similarity(indexes["Q"], text, rows_all),
    }


def main() -> None:
    """Print the diagnostic tables."""
    if not config.EVAL_FILE.exists():
        print("STOP: data/eval_set.csv is missing.")
        sys.exit(1)

    eval_df = pd.read_csv(config.EVAL_FILE)
    eval_df["should_escalate"] = eval_df["should_escalate"].astype(str).str.strip().str.lower()

    faq = pd.read_csv(config.FAQ_FILE).fillna("")
    indexes = {
        "QA": build_index(faq["question"] + " " + faq["answer"]),
        "Q": build_index(faq["question"]),
    }
    classifier, sentiment = ClassifierAgent(), SentimentAgent()

    # ---------- per-ticket table ----------
    rows = []
    for _, row in eval_df.iterrows():
        record = {"text": row["text"][:45], "should_escalate": row["should_escalate"],
                  "true_category": row["true_category"].split()[0]}
        scores = score_ticket(row["text"], classifier, sentiment, faq, indexes)
        scores["pred_category"] = scores["pred_category"].split()[0]
        record.update(scores)
        rows.append(record)
    table = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.RESULTS_DIR / "score_diagnostics.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("=== PER-TICKET SCORES (eval set) ===")
    print(table.round(2).to_string(index=False))

    # ---------- group summary ----------
    print("\n=== MIN / MEDIAN / MAX BY GROUP (no = routine, yes = should escalate) ===")
    columns = ["confidence", "sentiment"] + VARIANTS
    print(table.groupby("should_escalate")[columns].agg(["min", "median", "max"]).round(2).T.to_string())

    # ---------- threshold sweeps on ROUTINE tickets ----------
    routine = table[table["should_escalate"] == "no"]
    print(f"\n=== % OF THE {len(routine)} ROUTINE TICKETS THAT A THRESHOLD WOULD WRONGLY ESCALATE ===")
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sweep = pd.DataFrame(
        {variant: [(routine[variant] < t).mean() * 100 for t in thresholds] for variant in VARIANTS},
        index=[f"FAQ sim < {t:.2f}" for t in thresholds],
    )
    print(sweep.round(0).to_string())

    conf_thresholds = [0.30, 0.40, 0.50, 0.60]
    conf_sweep = pd.Series(
        [(routine["confidence"] < t).mean() * 100 for t in conf_thresholds],
        index=[f"confidence < {t:.2f}" for t in conf_thresholds],
    )
    print("\n(same idea for classifier confidence)")
    print(conf_sweep.round(0).to_string())

    # ---------- the three smoke-test tickets ----------
    print("\n=== THE THREE SMOKE-TEST TICKETS ===")
    for text in STANDARD_TICKETS:
        scores = score_ticket(text, classifier, sentiment, faq, indexes)
        print(f"\n{text}")
        for key, value in scores.items():
            print(f"    {key}: {round(value, 3) if isinstance(value, float) else value}")

    print(f"\nSaved: results\\score_diagnostics.csv")


if __name__ == "__main__":
    main()