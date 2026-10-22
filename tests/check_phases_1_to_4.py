import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

results = []   # list of (name, passed)


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def part_a_data() -> None:
    print("\n=== PART A: PHASE 1 DATA ===")
    files_ok = all(p.exists() for p in [config.TRAIN_FILE, config.TEST_FILE])
    check("train.csv and test.csv exist", files_ok)
    if not files_ok:
        return

    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    print(f"      train shape {train.shape} | test shape {test.shape}")

    check("'text' and 'category' columns present",
          {"text", "category"}.issubset(train.columns) and {"text", "category"}.issubset(test.columns))
    check("exactly our 5 categories in train and test",
          set(train["category"]) == set(config.CATEGORIES) and set(test["category"]) == set(config.CATEGORIES))

    train_counts = train["category"].value_counts()
    test_counts = test["category"].value_counts()
    check("train classes balanced (max - min <= 2)",
          train_counts.max() - train_counts.min() <= 2, str(train_counts.to_dict()))
    check("test classes balanced (max - min <= 2)",
          test_counts.max() - test_counts.min() <= 2, str(test_counts.to_dict()))

    test_share = len(test) / (len(train) + len(test))
    check("test share is about 20%", 0.19 <= test_share <= 0.21, f"{test_share:.3f}")
    check("no identical text in both train and test",
          len(set(train["text"]) & set(test["text"])) == 0)


def part_b_agents():
    print("\n=== PART B: AGENTS ===")
    from agents.classifier import ClassifierAgent
    from agents.sentiment import SentimentAgent
    from agents.knowledge import KnowledgeAgent
    from sklearn.metrics import accuracy_score

    # --- Phase 2: classifier ---
    classifier = ClassifierAgent()
    check("model classes equal our 5 categories",
          sorted(classifier.model.classes_) == sorted(config.CATEGORIES))
    test = pd.read_csv(config.TEST_FILE)
    accuracy = accuracy_score(test["category"], classifier.model.predict(test["text"]))
    check("classifier test accuracy >= 0.70", accuracy >= 0.70, f"accuracy {accuracy:.4f}")
    output = classifier.run({"text": "I want a refund for my last payment"})
    check("classifier output has category, confidence, priority",
          {"category", "confidence", "priority"} == set(output.keys()), str(output))

    # --- Phase 3: sentiment ---
    sentiment = SentimentAgent()
    angry = sentiment.run({"text": "I am extremely angry. Your service is terrible and I was charged twice!"})
    polite = sentiment.run({"text": "Thank you so much for your help, I really appreciate it!"})
    check("sentiment: angry text scores negative", angry["sentiment_score"] < -0.05, str(angry))
    check("sentiment: polite text scores positive", polite["sentiment_score"] > 0.05, str(polite))

    # --- Phase 4: knowledge ---
    faq = pd.read_csv(config.FAQ_FILE)
    check("faq.csv has 5 categories and 20+ entries",
          set(faq["category"]) == set(config.CATEGORIES) and len(faq) >= 20, f"{len(faq)} rows")
    knowledge = KnowledgeAgent()
    match = knowledge.run({"text": "How do I change my password?", "category": "Account & Subscription"})
    check("knowledge: password question matches the password FAQ",
          "password" in match["faq_question"].lower(), match["faq_question"])
    nonsense = knowledge.run({"text": "purple banana helicopter xyz", "category": "General Inquiry & Feedback"})
    check("knowledge: gibberish has similarity below 0.30",
          nonsense["faq_similarity"] < 0.30, str(nonsense["faq_similarity"]))

    return classifier, sentiment, knowledge


def part_c_chain(classifier, sentiment, knowledge) -> None:
    """Pass three tickets through all three agents using only the ticket dictionary."""
    print("\n=== PART C: THREE TICKETS THROUGH THE CHAIN ===")
    tickets = [
        "I was charged twice and nobody is helping me! THIS IS UNACCEPTABLE!!!",
        "How long will delivery take for my order?",
        "my thing is not working",
    ]
    for text in tickets:
        ticket = {"text": text}
        for agent in (classifier, sentiment, knowledge):
            ticket.update(agent.run(ticket))          # each agent adds its own fields
        print(f"\nTicket : {text}")
        print(f"  category   : {ticket['category']} (confidence {ticket['confidence']}, priority {ticket['priority']})")
        print(f"  sentiment  : {ticket['sentiment_score']} ({ticket['sentiment_label']})")
        print(f"  FAQ match  : {ticket['faq_question']}")
        print(f"  similarity : {ticket['faq_similarity']} (scope: {ticket['faq_scope']})")

    required = {"category", "confidence", "priority", "sentiment_score",
                "sentiment_label", "faq_question", "faq_answer", "faq_similarity"}
    check("final ticket contains every field from all three agents", required.issubset(ticket.keys()))


def main() -> None:
    part_a_data()
    try:
        agents = part_b_agents()
        part_c_chain(*agents)
    except Exception as error:      # a missing file or import problem
        check("agents loaded without error", False, f"{type(error).__name__}: {error}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n=== SUMMARY: {passed}/{len(results)} checks passed ===")
    if passed == len(results):
        print("PHASES 1-4: ALL PASS")
    else:
        print("PHASES 1-4: SOMETHING FAILED. Paste this whole output in the chat.")
        sys.exit(1)


if __name__ == "__main__":
    main()