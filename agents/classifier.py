from pathlib import Path

import joblib

import config


class ClassifierAgent:

    name = "Classifier Agent"

    def __init__(self, model_path: Path = None) -> None:
        path = Path(model_path) if model_path else config.MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python scripts\\train_classifier.py"
            )
        self.model = joblib.load(path)

    def run(self, ticket: dict) -> dict:
        text = str(ticket.get("text", "")).strip()

        # predict_proba gives one probability per category (they sum to 1).
        probabilities = self.model.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        category = str(self.model.classes_[best_index])

        return {
            "category": category,
            "confidence": round(float(probabilities[best_index]), 4),
            "priority": config.PRIORITY_MAP.get(category, "Medium"),
        }