import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
SEED = 42                      
TEST_SIZE = 0.20              
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "bitext_raw.csv"
TRAIN_FILE = ROOT / "data" / "train.csv"
TEST_FILE = ROOT / "data" / "test.csv"

CATEGORY_MAP = {
    "Billing & Refund": ["REFUND", "PAYMENT", "INVOICE"],
    "Order & Cancellation": ["ORDER", "CANCEL"],
    "Shipping & Delivery": ["DELIVERY", "SHIPPING"],
    "Account & Subscription": ["ACCOUNT", "SUBSCRIPTION"],
    "General Inquiry & Feedback": ["CONTACT", "FEEDBACK"],
}

EXPECTED_COLUMNS = ["instruction", "category", "intent", "response"]


def build_reverse_map() -> dict:
    """Turn CATEGORY_MAP into {source_category: our_category} for fast lookup."""
    reverse = {}
    for our_category, source_list in CATEGORY_MAP.items():
        for source in source_list:
            reverse[source] = our_category
    return reverse


def download_dataset() -> pd.DataFrame:
    print(f"Downloading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    print(f"Saved raw copy to: {RAW_FILE.relative_to(ROOT)}")
    return df


def main() -> None:
    # ---------- 1. Download and inspect ----------
    df = download_dataset()

    print("\n=== RAW DATA ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # Verify the columns we need really exist (do not guess!)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"\nSTOP: these expected columns are missing: {missing_cols}")
        print("Paste the 'Columns:' line above in the chat and I will adapt the code.")
        sys.exit(1)

    print("\nOriginal category counts:")
    print(df["category"].value_counts().to_string())

    # ---------- 2. Check the category mapping against real values ----------
    reverse_map = build_reverse_map()
    actual_categories = set(df["category"].astype(str).str.upper().unique())
    expected_sources = set(reverse_map.keys())

    not_found = sorted(expected_sources - actual_categories)
    unmapped = sorted(actual_categories - expected_sources)

    print("\nSource categories in my mapping but NOT in the dataset:", not_found)
    print("Dataset categories not in my mapping (will be dropped):", unmapped)

    if not_found:
        print("\nSTOP: the mapping does not match the real category names.")
        print("Paste everything printed above in the chat and I will fix the mapping.")
        sys.exit(1)

    # ---------- 3. Map to our five categories ----------
    df = df[["instruction", "category", "intent"]].copy()
    df["source_category"] = df["category"].astype(str).str.upper()
    df["category"] = df["source_category"].map(reverse_map)

    before_drop = len(df)
    df = df.dropna(subset=["category"])          # drop unmapped categories
    print(f"\nDropped {before_drop - len(df)} rows with unmapped categories.")

    # Clean the text and remove duplicate tickets 
    df = df.rename(columns={"instruction": "text"})
    df["text"] = df["text"].astype(str).str.strip()
    before_dedup = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"Removed {before_dedup - len(df)} duplicate ticket texts.")

    print("\n=== COUNTS PER CATEGORY: BEFORE BALANCING ===")
    print(df["category"].value_counts().to_string())

    # ---------- 4. Balance: same number of rows in every category ----------
    smallest = df["category"].value_counts().min()
    parts = []
    for category_name in CATEGORY_MAP:
        group = df[df["category"] == category_name]
        parts.append(group.sample(n=smallest, random_state=SEED))
    balanced = pd.concat(parts).sample(frac=1, random_state=SEED)  # shuffle
    balanced = balanced.reset_index(drop=True)

    print(f"\n=== COUNTS PER CATEGORY: AFTER BALANCING (each = {smallest}) ===")
    print(balanced["category"].value_counts().to_string())

    # ---------- 5. Stratified 80/20 split ----------
    train_df, test_df = train_test_split(
        balanced,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=balanced["category"],
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\n=== SPLIT ===")
    print("train shape:", train_df.shape)
    print("test shape :", test_df.shape)
    print("\nTrain counts per category:")
    print(train_df["category"].value_counts().to_string())
    print("\nTest counts per category:")
    print(test_df["category"].value_counts().to_string())

    print("\nSample rows from train.csv:")
    print(train_df.head(3).to_string())
    print(f"\nSaved: {TRAIN_FILE.relative_to(ROOT)} and {TEST_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()