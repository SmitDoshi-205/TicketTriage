import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.knowledge import KnowledgeAgent

LOW_SIMILARITY_LIMIT = 0.30   # the threshold we plan to use in Phase 5

# (query, category given, text that should appear in the matched FAQ question)
QUERIES = [
    ("I would like a refund for my order, please return my money",
     "Billing & Refund", "refund"),
    ("My package has not arrived yet, where is it?",
     "Shipping & Delivery", "not arrived"),
    ("How do I change my password?",
     "Account & Subscription", "password"),
    ("I need to cancel the order I placed yesterday",
     "Order & Cancellation", "cancel"),
    ("I want to talk to a real person",
     "General Inquiry & Feedback", "real person"),
]
GIBBERISH = ("purple banana helicopter xyz", "General Inquiry & Feedback")


def main() -> None:
    agent = KnowledgeAgent()
    passed = 0

    print("=== 5 REAL QUERIES ===")
    for text, category, keyword in QUERIES:
        result = agent.run({"text": text, "category": category})
        ok = keyword in result["faq_question"].lower()
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] query   : {text}")
        print(f"       matched : {result['faq_question']}")
        print(f"       similarity {result['faq_similarity']} | search scope: {result['faq_scope']}")

    print("\n=== GIBBERISH QUERY (similarity should be low) ===")
    result = agent.run({"text": GIBBERISH[0], "category": GIBBERISH[1]})
    low_ok = result["faq_similarity"] < LOW_SIMILARITY_LIMIT
    print(f"[{'PASS' if low_ok else 'FAIL'}] similarity {result['faq_similarity']} "
          f"(must be below {LOW_SIMILARITY_LIMIT})")

    print("\n=== FALLBACK TEST (unknown category -> search all FAQs) ===")
    result = agent.run({"text": "How do I reset my password?", "category": "Unknown"})
    fallback_ok = result["faq_scope"] == "all" and "password" in result["faq_question"].lower()
    print(f"[{'PASS' if fallback_ok else 'FAIL'}] scope = {result['faq_scope']}, "
          f"matched: {result['faq_question']}")

    if passed == len(QUERIES) and low_ok and fallback_ok:
        print("\nPHASE 4 CHECKPOINT: PASS")
    else:
        print("\nPHASE 4 CHECKPOINT: FAIL (paste the output in chat)")


if __name__ == "__main__":
    main()