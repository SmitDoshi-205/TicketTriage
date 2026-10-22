import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.sentiment import SentimentAgent

SAMPLES = [
    ("calm",             "Hello, I would like to know when my order will arrive."),
    ("angry, CAPS",      "THIS IS RIDICULOUS!!! I WANT MY MONEY BACK NOW!!!"),
    ("angry, plain",     "I am extremely angry. Your service is terrible and I was charged twice!"),
    ("polite",           "Thank you so much for your help, I really appreciate it!"),
    ("polite complaint", "Could you please check why I was charged twice? I would appreciate your help."),
    ("sarcastic 1",      "Oh great, another wonderful delivery delay. Thanks a lot for nothing."),
    ("sarcastic 2",      "Wow, fantastic service, my package only took three weeks."),
    ("vague",            "my thing is not working"),
]


def main() -> None:
    agent = SentimentAgent()
    scores = {}

    print(f"{'type':<17} {'score':>7}  {'label':<14} text")
    print("-" * 90)
    for kind, text in SAMPLES:
        result = agent.run({"text": text})
        scores[kind] = result["sentiment_score"]
        print(f"{kind:<17} {result['sentiment_score']:>7}  "
              f"{result['sentiment_label']:<14} {text}")

    # Sanity checks that any working sentiment tool should satisfy.
    check_1 = scores["angry, plain"] < scores["calm"]
    check_2 = scores["polite"] > 0.05
    print("\nangry scores lower than calm :", "PASS" if check_1 else "FAIL")
    print("polite scores positive       :", "PASS" if check_2 else "FAIL")

    if check_1 and check_2:
        print("PHASE 3 CHECKPOINT: PASS")
    else:
        print("PHASE 3 CHECKPOINT: FAIL (paste the output in chat)")


if __name__ == "__main__":
    main()