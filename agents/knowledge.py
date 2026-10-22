import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


class KnowledgeAgent:

    name = "Knowledge Agent"

    def __init__(self, faq_path=None) -> None:
        """Load faq.csv and build the TF-IDF index over the FAQ questions."""
        path = faq_path or config.FAQ_FILE
        self.faq = pd.read_csv(path).fillna("")

        needed = {"category", "question", "answer"}
        if not needed.issubset(self.faq.columns):
            raise ValueError(
                f"faq.csv must have columns {needed}, found {list(self.faq.columns)}"
            )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.faq_matrix = self.vectorizer.fit_transform(self.faq["question"])

    def _best_match(self, query_vector, rows: list):
        similarities = cosine_similarity(query_vector, self.faq_matrix[rows])[0]
        position = int(similarities.argmax())
        return rows[position], float(similarities[position])

    def run(self, ticket: dict) -> dict:
        text = str(ticket.get("text", ""))
        category = ticket.get("category")

        all_rows = self.faq.index.tolist()
        category_rows = self.faq.index[self.faq["category"] == category].tolist()
        query_vector = self.vectorizer.transform([text])

        if category_rows:
            best_row, best_similarity = self._best_match(query_vector, category_rows)
            scope = "category"
        else:
            best_row, best_similarity = self._best_match(query_vector, all_rows)
            scope = "all"

        if scope == "category" and best_similarity < config.FAQ_FALLBACK_THRESHOLD:
            other_row, other_similarity = self._best_match(query_vector, all_rows)
            if other_similarity > best_similarity:
                best_row, best_similarity, scope = other_row, other_similarity, "all_fallback"

        return {
            "faq_question": str(self.faq.loc[best_row, "question"]),
            "faq_answer": str(self.faq.loc[best_row, "answer"]),
            "faq_similarity": round(best_similarity, 4),
            "faq_scope": scope,
        }