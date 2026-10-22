from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from orchestrator import Orchestrator

app = FastAPI(title="TicketTriage API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

orch = Orchestrator(enable_logging=False)

class Query(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "I was charged twice for my order"}
        }
    )

    text: str = Field(..., min_length=1, description="Customer support ticket text")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


@app.get("/health")
def health():
    return {"status": "ok", "agents": [a.name for a in orch.pipeline]}

@app.post("/triage")
def triage(q: Query):
    return orch.process(q.text, log=False)