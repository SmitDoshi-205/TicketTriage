import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score

import config
from agents.classifier import ClassifierAgent

# (sentence, category we hope for)
SAMPLES = [
    ("I want a refund for my last payment", "Billing & Refund"),
    ("Please cancel my order number 12345", "Order & Cancellation"),
    ("How long will delivery take for my order?", "Shipping & Delivery"),
    ("I forgot my password and cannot log in to my account", "Account & Subscription"),
    ("How can I contact customer service?", "General Inquiry & Feedback"),
]


def main() -> None:
    agent = ClassifierAgent()

    print("=== 5 SAMPLE SENTENCES ===")
    matches = 0
    for text, expected in SAMPLES:
        result = agent.run({"text": text})
        ok = result["category"] == expected
        matches += ok
        print(f"[{'MATCH' if ok else 'DIFF '}] {text}")
        print(f"        -> {result['category']} | confidence {result['confidence']} "
              f"| priority {result['priority']}")
        if not ok:
            print(f"        (I expected: {expected})")
    print(f"\nSample matches: {matches}/{len(SAMPLES)}")

    # Batch prediction on the whole test split (fast).
    test_df = pd.read_csv(config.TEST_FILE)
    predictions = agent.model.predict(test_df["text"])
    accuracy = accuracy_score(test_df["category"], predictions)
    print(f"\nTest accuracy using the saved model: {accuracy:.4f}")

    if accuracy >= 0.70:
        print("PHASE 2 CHECKPOINT: PASS")
    else:
        print("PHASE 2 CHECKPOINT: FAIL (accuracy unexpectedly low, paste output in chat)")


if __name__ == "__main__":
    main()