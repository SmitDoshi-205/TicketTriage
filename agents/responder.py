import config

# Opening sentence chosen by how upset the customer sounds.
EMPATHY = {
    "Very Negative": "I am truly sorry for the frustration this has caused you, and I completely understand why you are upset.",
    "Negative": "I am sorry for the trouble you have faced.",
    "Neutral": "Thank you for reaching out, and sorry for any inconvenience.",
    "Positive": "Thank you for reaching out, and sorry for any inconvenience.",
}

# Extra sentence for special escalation reasons.
SPECIAL_LINES = {
    "LEGAL_FRAUD": "We take legal and fraud-related concerns very seriously, so your message has been marked urgent and sent to a specialist team.",
    "LOW_FAQ_SIMILARITY": "We could not find a reliable automatic answer to your question, so a person will look at it directly.",
    "LOW_CONFIDENCE": "We want to be sure we understand your problem correctly, so a person will read your message directly.",
}


class ResponderAgent:

    name = "Responder Agent"

    def run(self, ticket: dict) -> dict:
       
        decision = ticket.get("decision", "ESCALATE")
        code = ticket.get("reason_code", "")
        score = ticket.get("sentiment_score", 0.0)
        upset = score <= config.SENTIMENT_ANGRY

        if decision == "ESCALATE" and code == "LOW_CONFIDENCE" and not upset:
            return {"reply": self._more_details_reply(), "reply_type": "MORE_DETAILS"}
        if decision == "ESCALATE":
            return {"reply": self._escalation_reply(ticket), "reply_type": "ESCALATION"}
        return {"reply": self._auto_reply(ticket), "reply_type": "AUTO_RESOLVE"}

    @staticmethod
    def _auto_reply(ticket: dict) -> str:
        topic = ticket.get("category", "your request")
        return (
            "Dear Customer,\n\n"
            f"Thank you for contacting us about {topic}.\n\n"
            f"{ticket.get('faq_answer', '')}\n\n"
            "If this does not fully solve your problem, just reply to this message "
            "and a member of our team will help you further.\n\n"
            f"Regards,\n{config.SUPPORT_TEAM_NAME}"
        )

    @staticmethod
    def _escalation_reply(ticket: dict) -> str:
        label = ticket.get("sentiment_label", "Neutral")
        code = ticket.get("reason_code", "")
        topic = ticket.get("category")
        subject = f"Your message about {topic}" if topic else "Your message"

        empathy = EMPATHY.get(label, EMPATHY["Neutral"])
        special = SPECIAL_LINES.get(code, "")
        passed_on = (
            f"{subject} has been passed to a senior member of our support team. "
            f"A human agent will contact you {config.HUMAN_CONTACT_PROMISE}."
        )

        # Interim help: only when the FAQ match is reliable and the ticket is
        # not a legal/fraud matter (where a generic FAQ answer would be wrong).
        interim = ""
        reliable = ticket.get("faq_similarity", 0.0) >= config.FAQ_SIMILARITY_THRESHOLD
        if code != "LEGAL_FRAUD" and reliable and ticket.get("faq_answer"):
            interim = (
                f"In the meantime, this may help ({ticket.get('faq_question', '')}):\n"
                f"{ticket['faq_answer']}"
            )

        parts = ["Dear Customer,", empathy, passed_on, special, interim,
                 "Thank you for your patience.",
                 f"Regards,\n{config.SUPPORT_TEAM_NAME}"]
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _more_details_reply() -> str:
        return (
            "Dear Customer,\n\n"
            "Thank you for reaching out. We want to help, but we could not fully "
            "understand your request.\n\n"
            "Could you please reply with a few more details? For example:\n"
            "- What exactly is the problem?\n"
            "- Your order number (if it is about an order or a payment)\n"
            "- When did it happen?\n\n"
            "A member of our team will also review your message.\n\n"
            f"Regards,\n{config.SUPPORT_TEAM_NAME}"
        )