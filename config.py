from pathlib import Path

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------
# Paths (all relative to the project root, never absolute)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
FAQ_FILE = DATA_DIR / "faq.csv"
EVAL_FILE = DATA_DIR / "eval_set.csv"
MODEL_FILE = ROOT / "models" / "classifier.joblib"
RESULTS_DIR = ROOT / "results"
LOG_FILE = RESULTS_DIR / "ticket_log.csv"

# ----------------------------------------------------------------------
# Our five categories and their priorities
# ----------------------------------------------------------------------
CATEGORIES = [
    "Billing & Refund",
    "Order & Cancellation",
    "Shipping & Delivery",
    "Account & Subscription",
    "General Inquiry & Feedback",
]

PRIORITY_MAP = {
    "Billing & Refund": "High",
    "Order & Cancellation": "Medium",
    "Shipping & Delivery": "Medium",
    "Account & Subscription": "Medium",
    "General Inquiry & Feedback": "Low",
}

# ----------------------------------------------------------------------
# Sentiment LABEL thresholds (used only to name the score; applied to the
# VADER compound score)
#   score <= SENTIMENT_VERY_NEGATIVE -> "Very Negative"
#   score <= SENTIMENT_NEGATIVE      -> "Negative"
#   score <  SENTIMENT_POSITIVE      -> "Neutral"
#   otherwise                        -> "Positive"
# ----------------------------------------------------------------------
SENTIMENT_VERY_NEGATIVE = -0.60
SENTIMENT_NEGATIVE = -0.05
SENTIMENT_POSITIVE = 0.05

# ----------------------------------------------------------------------
# KNOWLEDGE AGENT SETTINGS
# The agent searches the predicted category first. If its best match there
# has a similarity below FAQ_FALLBACK_THRESHOLD, it also searches ALL FAQs
# and keeps whichever match is better (protects against classifier mistakes).
# ----------------------------------------------------------------------
FAQ_FALLBACK_THRESHOLD = 0.30

# ----------------------------------------------------------------------
# ESCALATION RULES (checked in this order by the Escalation Agent)
#   1. legal / fraud keyword found                 -> ESCALATE
#   2. classifier confidence  <  CONFIDENCE_THRESHOLD       -> ESCALATE
#   3. FAQ similarity         <  FAQ_SIMILARITY_THRESHOLD   -> ESCALATE
#   4. sentiment score        <= SENTIMENT_VERY_ANGRY       -> ESCALATE
#   5. sentiment score        <= SENTIMENT_ANGRY and priority == "High"
#                                                           -> ESCALATE
#   otherwise -> AUTO_RESOLVE
# ----------------------------------------------------------------------
LEGAL_FRAUD_KEYWORDS = [
    "lawyer", "legal action", "legal notice", "lawsuit", "sue", "suing",
    "consumer court", "consumer forum", "chargeback",
    "fraud", "fraudulent", "scam", "police",
]

CONFIDENCE_THRESHOLD = 0.50
FAQ_SIMILARITY_THRESHOLD = 0.30
SENTIMENT_VERY_ANGRY = -0.60
SENTIMENT_ANGRY = -0.30
HIGH_PRIORITY = "High"

# ----------------------------------------------------------------------
# Reply template settings (demo wording, not a real service promise)
# ----------------------------------------------------------------------
SUPPORT_TEAM_NAME = "Customer Support Team"
HUMAN_CONTACT_PROMISE = "within 24 hours"