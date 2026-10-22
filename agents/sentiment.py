from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config


class SentimentAgent:

    name = "Sentiment Agent"

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def label_for(score: float) -> str:
        if score <= config.SENTIMENT_VERY_NEGATIVE:
            return "Very Negative"
        if score <= config.SENTIMENT_NEGATIVE:
            return "Negative"
        if score < config.SENTIMENT_POSITIVE:
            return "Neutral"
        return "Positive"

    def run(self, ticket: dict) -> dict:
        text = str(ticket.get("text", ""))
        score = self.analyzer.polarity_scores(text)["compound"]
        return {
            "sentiment_score": round(float(score), 4),
            "sentiment_label": self.label_for(score),
        }