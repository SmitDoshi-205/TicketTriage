import csv
import time
import uuid
from datetime import datetime
from pathlib import Path

import config
from agents.classifier import ClassifierAgent
from agents.escalation import EscalationAgent
from agents.knowledge import KnowledgeAgent
from agents.responder import ResponderAgent
from agents.sentiment import SentimentAgent

# Columns written to the log CSV (one row per ticket).
LOG_COLUMNS = [
    "ticket_id", "created_at", "text", "category", "confidence", "priority",
    "sentiment_score", "sentiment_label", "faq_question", "faq_similarity",
    "decision", "reason", "reply_type", "reply", "total_time_ms",
]


class Orchestrator:

    def __init__(self, enable_logging: bool = True, log_file=None) -> None:
        self.classifier = ClassifierAgent()
        self.sentiment = SentimentAgent()
        self.knowledge = KnowledgeAgent()
        self.escalation = EscalationAgent()
        self.responder = ResponderAgent()

        # The order matters: later agents read what earlier agents wrote.
        self.pipeline = [
            self.classifier, self.sentiment, self.knowledge,
            self.escalation, self.responder,
        ]
        self.enable_logging = enable_logging
        self.log_file = Path(log_file) if log_file else config.LOG_FILE

    def process(self, text: str, log: bool = None) -> dict:
        
        ticket = {
            "ticket_id": uuid.uuid4().hex[:8],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "text": str(text).strip(),
        }
        trace = []
        timings = {}

        pipeline_start = time.perf_counter()
        for agent in self.pipeline:
            step_start = time.perf_counter()
            output = agent.run(ticket)                      # agent returns only its own fields
            elapsed_ms = round((time.perf_counter() - step_start) * 1000, 2)

            ticket.update(output)                           # merge into the shared ticket
            trace.append({"agent": agent.name, "output": output, "time_ms": elapsed_ms})
            timings[agent.name] = elapsed_ms

        ticket["trace"] = trace
        ticket["timings_ms"] = timings
        ticket["total_time_ms"] = round((time.perf_counter() - pipeline_start) * 1000, 2)

        should_log = self.enable_logging if log is None else log
        if should_log:
            self._append_log(ticket)
        return ticket

    def process_many(self, texts, log: bool = None) -> list:
        return [self.process(text, log=log) for text in texts]

    def _append_log(self, ticket: dict) -> None:
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            write_header = (not self.log_file.exists()) or self.log_file.stat().st_size == 0
            with open(self.log_file, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(ticket)
        except PermissionError:
            print(f"WARNING: could not write to {self.log_file}. Close it in Excel and retry.")