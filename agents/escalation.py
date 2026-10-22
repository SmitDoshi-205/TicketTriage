import re

import config


class EscalationAgent:

    name = "Escalation Agent"

    def __init__(self) -> None:

        pattern = r"\b(?:" + "|".join(re.escape(k) for k in config.LEGAL_FRAUD_KEYWORDS) + r")\b"
        self.keyword_regex = re.compile(pattern, re.IGNORECASE)

    @staticmethod
    def _escalate(code: str, reason: str) -> dict:
        return {"decision": "ESCALATE", "reason": reason, "reason_code": code}

    def run(self, ticket: dict) -> dict:
        
        text = str(ticket.get("text", ""))

        # Rule 1: legal or fraud keywords
        match = self.keyword_regex.search(text)
        if match:
            return self._escalate(
                "LEGAL_FRAUD",
                f"Legal or fraud keyword detected: '{match.group(0).lower()}'",
            )

        # Rule 2: the classifier is unsure what the ticket is about
        if "confidence" in ticket and ticket["confidence"] < config.CONFIDENCE_THRESHOLD:
            return self._escalate(
                "LOW_CONFIDENCE",
                f"Unable to understand the request (classifier confidence "
                f"{ticket['confidence']:.2f} < {config.CONFIDENCE_THRESHOLD:.2f})",
            )

        # Rule 3: no FAQ answer is close enough to trust
        if "faq_similarity" in ticket and ticket["faq_similarity"] < config.FAQ_SIMILARITY_THRESHOLD:
            return self._escalate(
                "LOW_FAQ_SIMILARITY",
                f"No reliable answer found (FAQ similarity "
                f"{ticket['faq_similarity']:.2f} < {config.FAQ_SIMILARITY_THRESHOLD:.2f})",
            )

        # Rule 4: very angry customer
        score = ticket.get("sentiment_score")
        if score is not None and score <= config.SENTIMENT_VERY_ANGRY:
            return self._escalate(
                "VERY_ANGRY",
                f"Very angry customer (sentiment {score:.2f} <= {config.SENTIMENT_VERY_ANGRY:.2f})",
            )

        # Rule 5: unhappy customer AND a high-priority category
        if (score is not None
                and score <= config.SENTIMENT_ANGRY
                and ticket.get("priority") == config.HIGH_PRIORITY):
            return self._escalate(
                "ANGRY_HIGH_PRIORITY",
                f"Unhappy customer on a high-priority issue (sentiment {score:.2f} "
                f"<= {config.SENTIMENT_ANGRY:.2f}, priority {config.HIGH_PRIORITY})",
            )

        # No rule fired: the ticket is routine.
        return {
            "decision": "AUTO_RESOLVE",
            "reason": "Routine ticket: clear category, reliable FAQ answer, no strong negative sentiment",
            "reason_code": "ROUTINE",
        }