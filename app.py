import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

import config
from orchestrator import Orchestrator

st.set_page_config(page_title="Ticket Triage", layout="wide")

MAX_BATCH_ROWS = 500          
GREEN = "#1e8e3e"
RED = "#d93025"

EXAMPLES = [
    ("Angry billing",
     "This is the WORST service ever! I was charged twice for my order and I am FURIOUS. Fix this NOW!!!"),
    ("Routine shipping", "How long will delivery take for my order?"),
    ("Vague message", "my thing is not working"),
    ("Legal threat",
     "I was charged twice and nobody replied. I will go to consumer court if this is not fixed today."),
    ("Password reset", "i forgot my password, how do i reset it"),
]

RESULT_TABLES = [
    ("Baseline A vs full multi-agent system", "baseline_comparison.csv"),
    ("Escalation metrics (full system)", "escalation_metrics.csv"),
    ("Ablation: one agent removed at a time", "ablation.csv"),
    ("Classifier on the Bitext test split", "classifier_test_metrics.csv"),
    ("Classifier on the messy eval set", "classifier_eval_metrics.csv"),
    ("Average processing time per agent", "timing.csv"),
]
RESULT_IMAGES = [
    ("Confusion matrix: Bitext test split", "confusion_matrix_test.png"),
    ("Confusion matrix: messy eval set", "confusion_matrix_eval.png"),
    ("Baseline A vs multi-agent system", "baseline_vs_multiagent.png"),
    ("Decisions by reason", "decision_distribution.png"),
]


@st.cache_resource(show_spinner="Loading models...")
def load_orchestrator() -> Orchestrator:
    return Orchestrator(enable_logging=True)


def set_example(text: str) -> None:
    st.session_state["ticket_text"] = text


def decision_badge(decision: str) -> str:
    colour = GREEN if decision == "AUTO_RESOLVE" else RED
    return (f'<span style="background-color:{colour};color:white;padding:6px 18px;'
            f'border-radius:6px;font-weight:bold;font-size:1.3rem;">{decision}</span>')


def render_result(ticket: dict) -> None:
    st.markdown(decision_badge(ticket["decision"]), unsafe_allow_html=True)
    st.write(f"**Reason:** {ticket['reason']}")
    st.write(f"**Category:** {ticket['category']}")

    columns = st.columns(4)
    columns[0].metric("Classifier confidence", f"{ticket['confidence']:.2f}")
    columns[1].metric("Priority", ticket["priority"])
    columns[2].metric(f"Sentiment ({ticket['sentiment_label']})", f"{ticket['sentiment_score']:+.2f}")
    columns[3].metric("FAQ similarity", f"{ticket['faq_similarity']:.2f}")
    st.write(f"**Matched FAQ:** {ticket['faq_question']}")

    st.subheader("Agent trace")
    st.caption(f"Each agent reads the shared ticket and adds its own fields. "
               f"Total pipeline time: {ticket['total_time_ms']} ms.")
    for number, step in enumerate(ticket["trace"], start=1):
        title = f"Step {number}: {step['agent']}  ({step['time_ms']} ms)"
        with st.expander(title, expanded=(step["agent"] != "Responder Agent")):
            st.json(step["output"])

    st.subheader("Drafted reply")
    st.caption(f"Template used: {ticket['reply_type']}")
    st.text_area("Drafted reply", ticket["reply"], height=320, disabled=True,
                 label_visibility="collapsed")


def run_batch(texts: pd.Series, orchestrator: Orchestrator, save_to_log: bool) -> pd.DataFrame:
    rows = []
    progress = st.progress(0.0)
    for position, text in enumerate(texts, start=1):
        ticket = orchestrator.process(text, log=save_to_log)
        rows.append({
            "ticket_id": ticket["ticket_id"],
            "text": ticket["text"],
            "category": ticket["category"],
            "confidence": ticket["confidence"],
            "priority": ticket["priority"],
            "sentiment_score": ticket["sentiment_score"],
            "sentiment_label": ticket["sentiment_label"],
            "faq_similarity": ticket["faq_similarity"],
            "decision": ticket["decision"],
            "reason": ticket["reason"],
            "reply_type": ticket["reply_type"],
            "reply": ticket["reply"],
        })
        progress.progress(position / len(texts))
    progress.empty()
    return pd.DataFrame(rows)


def read_uploaded_csv(uploaded) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded, encoding="utf-8-sig")
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="latin-1")


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.title("Multi-Agent Customer Support Ticket Triage")

try:
    orchestrator = load_orchestrator()
except FileNotFoundError as error:
    st.error(f"{error}")
    st.stop()

save_to_log = st.checkbox("Save processed tickets to results/ticket_log.csv", value=True, key="save_to_log")

tab_single, tab_batch, tab_metrics = st.tabs(["Single ticket", "Batch (CSV)", "Metrics"])

# ---------------- Tab 1: single ticket ----------------
with tab_single:
    st.write("Click an example or type your own ticket:")
    st.session_state.setdefault("ticket_text", "")

    example_columns = st.columns(len(EXAMPLES))
    for index, (label, example_text) in enumerate(EXAMPLES):
        example_columns[index].button(label, key=f"example_{index}",
                                      on_click=set_example, args=(example_text,))

    st.text_area("Customer ticket", key="ticket_text", height=130)
    analyse_clicked = st.button("Analyse ticket", key="analyse_button", type="primary")

    if analyse_clicked:
        ticket_text = st.session_state.get("ticket_text", "").strip()
        if not ticket_text:
            st.warning("Please type a ticket or click an example first.")
        else:
            with st.spinner("Running the five agents..."):
                st.session_state["result"] = orchestrator.process(ticket_text, log=save_to_log)

    if "result" in st.session_state:
        st.divider()
        render_result(st.session_state["result"])

# ---------------- Tab 2: batch ----------------
with tab_batch:
    st.write(f"Upload a CSV with one ticket per row (first {MAX_BATCH_ROWS} rows are processed). "
             "You can try `data/eval_set.csv`, which has a `text` column.")
    uploaded = st.file_uploader("CSV file", type="csv", key="batch_file")

    if uploaded is not None:
        try:
            batch_df = read_uploaded_csv(uploaded)
        except Exception as error:
            batch_df = None
            st.error(f"Could not read this file as a CSV: {error}")

        if batch_df is not None and batch_df.empty:
            st.warning("The CSV has no rows.")
        elif batch_df is not None:
            column_names = list(batch_df.columns)
            default_index = column_names.index("text") if "text" in column_names else 0
            text_column = st.selectbox("Column that contains the ticket text", column_names,
                                       index=default_index, key="batch_column")
            st.write(f"{len(batch_df)} rows found. Preview:")
            st.dataframe(batch_df.head(5), hide_index=True)

            if st.button("Process all tickets", key="batch_button", type="primary"):
                texts = batch_df[text_column].fillna("").astype(str).str.strip()
                texts = texts[texts != ""]
                if len(texts) > MAX_BATCH_ROWS:
                    st.warning(f"Only the first {MAX_BATCH_ROWS} tickets are processed.")
                    texts = texts.head(MAX_BATCH_ROWS)
                st.session_state["batch_result"] = run_batch(texts, orchestrator, save_to_log)

    if "batch_result" in st.session_state:
        batch_result = st.session_state["batch_result"]
        counts = batch_result["decision"].value_counts()
        summary = st.columns(3)
        summary[0].metric("Tickets processed", len(batch_result))
        summary[1].metric("AUTO_RESOLVE", int(counts.get("AUTO_RESOLVE", 0)))
        summary[2].metric("ESCALATE", int(counts.get("ESCALATE", 0)))
        st.dataframe(batch_result.drop(columns=["reply"]), hide_index=True)
        st.download_button(
            "Download results as CSV",
            data=batch_result.to_csv(index=False).encode("utf-8-sig"),
            file_name="triage_results.csv",
            mime="text/csv",
            key="batch_download",
        )

# ---------------- Tab 3: metrics ----------------
with tab_metrics:
    st.write("These tables and charts were produced by `python evaluate.py` and saved in `results/`.")
    shown_anything = False

    for title, file_name in RESULT_TABLES:
        path = config.RESULTS_DIR / file_name
        if path.exists():
            shown_anything = True
            st.subheader(title)
            st.dataframe(pd.read_csv(path), hide_index=True)

    image_columns = st.columns(2)
    visible = [(t, f) for t, f in RESULT_IMAGES if (config.RESULTS_DIR / f).exists()]
    for position, (title, file_name) in enumerate(visible):
        shown_anything = True
        image_columns[position % 2].image(str(config.RESULTS_DIR / file_name), caption=title)

    if not shown_anything:
        st.info("No saved results found. Run `python evaluate.py` first.")

    st.caption("Limitations: the training data is templated (Bitext), VADER misses sarcasm, and the "
               "evaluation set has only 50 hand-written tickets.")