import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, BaseMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from langchain_ollama import ChatOllama


# ----------------------------
# Configuration
# ----------------------------
os.environ.setdefault("USER_AGENT", "my-langchain-agent/1.0")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qc:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")  # optional
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

SYSTEM_PROMPT = (
    "You are a tool-using AI assistant.\n"
    "Use PythonREPLTool for any computation.\n"
    "Use DuckDuckGoSearchRun for any web/current info.\n"
    "If you did not call a tool, DO NOT claim you did.\n"
    "Be concise and correct."
)


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="LangChain Agent Microservice", version="1.0.0")

_agent = None  # initialized on startup


def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


def build_agent():
    tools = build_tools()

    model_kwargs = dict(model=OLLAMA_MODEL, temperature=TEMPERATURE)
    if OLLAMA_BASE_URL:
        model_kwargs["base_url"] = OLLAMA_BASE_URL

    model = ChatOllama(**model_kwargs)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,  # LangChain v1 create_agent uses system_prompt
    )
    return agent


@app.on_event("startup")
def startup():
    global _agent
    _agent = build_agent()


@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL}


# ----------------------------
# API Schemas
# ----------------------------
class InvokeRequest(BaseModel):
    query: str = Field(..., description="User query for the agent")
    include_trace: bool = Field(
        default=False,
        description="If true, return the full message list (useful for debugging)",
    )


class MessageOut(BaseModel):
    type: str
    content: str


class InvokeResponse(BaseModel):
    answer: str
    trace: Optional[List[MessageOut]] = None


def _serialize_messages(messages: List[BaseMessage]) -> List[MessageOut]:
    out: List[MessageOut] = []
    for m in messages:
        out.append(
            MessageOut(
                type=m.__class__.__name__,
                content=getattr(m, "content", "") or "",
            )
        )
    return out


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result: Dict[str, Any] = _agent.invoke(
            {"messages": [HumanMessage(content=req.query)]}
        )
        messages = result.get("messages", [])
        if not messages:
            raise RuntimeError("Agent returned no messages")

        answer = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

        if req.include_trace:
            return InvokeResponse(answer=answer, trace=_serialize_messages(messages))
        return InvokeResponse(answer=answer)

    except Exception as e:
        # You might want to log e with a proper logger in production
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {e}")


# ----------------------------
# Optional: simple SSE endpoint
# ----------------------------
class StreamRequest(BaseModel):
    query: str


@app.post("/invoke/stream")
def invoke_stream(req: StreamRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    def gen():
        yield "event: status\ndata: started\n\n"

        try:
            result: Dict[str, Any] = _agent.invoke(
                {"messages": [HumanMessage(content=req.query)]}
            )
            messages = result.get("messages", [])
            answer = messages[-1].content if messages else ""

            # Fix: sanitize before f-string
            cleaned = answer.replace("\n", "\\n")

            yield f"event: final\ndata: {cleaned}\n\n"

        except Exception as e:
            cleaned_err = str(e).replace("\n", "\\n")
            yield f"event: error\ndata: {cleaned_err}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
