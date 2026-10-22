import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import config


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.SEED,
        )),
    ])


def main() -> None:
    # ---------- load data ----------
    if not config.TRAIN_FILE.exists() or not config.TEST_FILE.exists():
        print("STOP: data/train.csv or data/test.csv is missing. Run Phase 1 first:")
        print("    python scripts\\prepare_data.py")
        sys.exit(1)

    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    print(f"Training rows: {len(train_df)} | Test rows: {len(test_df)}")

    # ---------- train ----------
    model = build_model()
    model.fit(train_df["text"], train_df["category"])

    # ---------- evaluate on data the model has never seen ----------
    predictions = model.predict(test_df["text"])
    y_true = test_df["category"]

    accuracy = accuracy_score(y_true, predictions)
    print(f"\nTEST ACCURACY: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    print("\nPer-category precision / recall / F1:")
    print(classification_report(y_true, predictions, digits=3))

    # ---------- confusion matrix ----------
    labels = config.CATEGORIES
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    short = [name.split()[0] for name in labels]          # Billing, Order, ...
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{s}" for s in short],
        columns=[f"pred_{s}" for s in short],
    )
    print("Confusion matrix (rows = true category, columns = predicted):")
    print(matrix_df.to_string())

    # ---------- save results and model ----------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    matrix_df.to_csv(config.RESULTS_DIR / "classifier_confusion_matrix.csv")
    report_dict = classification_report(y_true, predictions, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(
        config.RESULTS_DIR / "classifier_report.csv"
    )

    joblib.dump(model, config.MODEL_FILE)
    print(f"\nSaved model to: {config.MODEL_FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()