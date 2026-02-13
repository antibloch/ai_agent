import os
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, Any, Optional, Dict, List

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ============================================================
# 0) Tracing / Logging
# ============================================================
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _short(s: Any, n: int = 500) -> str:
    """Safe shortener for logs."""
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = s.replace("\n", "\\n")
    return s[:n] + ("…" if len(s) > n else "")


def _msg_role(m: BaseMessage) -> str:
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, HumanMessage):
        return "human"
    if isinstance(m, ToolMessage):
        return f"tool:{m.name}"
    if isinstance(m, AIMessage):
        return "ai"
    return m.__class__.__name__


def _serialize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    out = []
    for tc in tool_calls or []:
        out.append(
            {
                "id": tc.get("id"),
                "name": tc.get("name"),
                "args": tc.get("args"),
            }
        )
    return out


class TraceLogger:
    """
    Two sinks:
      - pretty console logs (logging)
      - optional JSONL file for programmatic analysis
    """

    def __init__(self, name: str = "agent_trace", jsonl_path: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.jsonl_path = jsonl_path

    def event(self, kind: str, data: Dict[str, Any]):
        payload = {"ts": _now_iso(), "kind": kind, **data}

        # Console: keep it readable
        self.logger.info(
            "[%s] %s",
            kind,
            _short(payload, 2000),
        )

        # JSONL: keep full structured data
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ----------------------------
# 1) Graph state
# ----------------------------
class AgentState(TypedDict, total=False):
    # Running transcript for the CURRENT attempt
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Orchestrator state
    original_query: str
    attempt: int
    max_attempts: int
    reflection: str  # critique text from previous attempt
    critique: str    # latest critique result

    # Tracing
    run_id: str


# ----------------------------
# 2) Tools + model
# ----------------------------
def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


tools = build_tools()
tools_by_name = {t.name: t for t in tools}

model = (
    ChatOllama(model="qc:latest", temperature=0)
    .bind_tools(tools)  # enables tool calling
)

BASE_SYSTEM_PROMPT = (
    "You are a tool-using AI assistant.\n"
    "Use PythonREPLTool for any computation.\n"
    "Use DuckDuckGoSearchRun for any web/current info.\n"
    "If you did not call a tool, DO NOT claim you did.\n"
    "Be concise and correct."
)

CRITIQUE_PROMPT = (
    "You are a strict reviewer.\n"
    "Critically analyze the assistant’s work using the FULL transcript.\n"
    "Check for:\n"
    "- Missing tool usage (if needed)\n"
    "- Incorrect computation\n"
    "- Outdated info\n"
    "- Hallucinations / claiming tools were used when they were not\n"
    "- Not following the requested format (e.g., not 1 line)\n\n"
    "If everything is correct, reply ONLY with: CORRECT.\n"
    "Otherwise explain briefly what is wrong."
)


# ----------------------------
# 3) Helpers
# ----------------------------
def _normalize_tool_args(args: Any):
    """
    Tool-call args can arrive as:
      - dict with "__arg1" for single-input tools
      - 1-key dict (e.g., {"query": "..."} or {"code": "..."})
      - raw string
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        if "__arg1" in args:
            return args["__arg1"]
        if len(args) == 1:
            return next(iter(args.values()))
        return args
    return args


def _system_message_for_attempt(state: AgentState) -> SystemMessage:
    """
    Base system prompt + optional reflection injected for retries.
    """
    sys = BASE_SYSTEM_PROMPT
    reflection = (state.get("reflection") or "").strip()
    if reflection:
        sys += f"\n\nPrevious failure notes:\n{reflection}\nFix the issues above."
    return SystemMessage(content=sys)


def _messages_snapshot(messages: Sequence[BaseMessage], max_items: int = 25) -> List[Dict[str, Any]]:
    """Serialize a bounded snapshot of messages for logging."""
    snap = []
    for m in list(messages)[-max_items:]:
        item = {
            "role": _msg_role(m),
            "content": getattr(m, "content", ""),
        }
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                item["tool_calls"] = _serialize_tool_calls(tool_calls)
        if isinstance(m, ToolMessage):
            item["tool_call_id"] = getattr(m, "tool_call_id", None)
        snap.append(item)
    return snap


# ============================================================
# 4) Nodes (with tracing)
# ============================================================
TRACE = TraceLogger(jsonl_path="agent_trace.jsonl")  # set None to disable file


def call_model(state: AgentState, config: RunnableConfig):
    """
    Tool-capable model call for the current attempt.
    """
    run_id = state.get("run_id", "unknown")
    attempt = int(state.get("attempt", 1))

    system_msg = _system_message_for_attempt(state)
    in_msgs = [system_msg] + list(state["messages"])

    TRACE.event(
        "llm.call.start",
        {
            "run_id": run_id,
            "attempt": attempt,
            "node": "agent",
            "input_messages": _messages_snapshot(in_msgs),
        },
    )

    t0 = time.perf_counter()
    response = model.invoke(in_msgs, config=config)
    dt_ms = int((time.perf_counter() - t0) * 1000)

    tool_calls = getattr(response, "tool_calls", None) or []
    TRACE.event(
        "llm.call.end",
        {
            "run_id": run_id,
            "attempt": attempt,
            "node": "agent",
            "latency_ms": dt_ms,
            "output_role": _msg_role(response),
            "output_content": getattr(response, "content", ""),
            "tool_calls": _serialize_tool_calls(tool_calls),
        },
    )

    return {"messages": [response]}


def tool_node(state: AgentState):
    """
    Execute tool_calls from the last AI message and return ToolMessage(s).
    """
    run_id = state.get("run_id", "unknown")
    attempt = int(state.get("attempt", 1))

    last = state["messages"][-1]
    outputs = []

    tool_calls = getattr(last, "tool_calls", None) or []
    TRACE.event(
        "tools.batch.start",
        {
            "run_id": run_id,
            "attempt": attempt,
            "node": "tools",
            "num_tool_calls": len(tool_calls),
            "tool_calls": _serialize_tool_calls(tool_calls),
        },
    )

    for tc in tool_calls:
        name = tc["name"]
        tool = tools_by_name.get(name)
        tool_call_id = tc.get("id", "")
        tool_args_raw = tc.get("args")
        tool_args = _normalize_tool_args(tool_args_raw)

        if tool is None:
            msg = ToolMessage(
                content=f"Unknown tool: {name}",
                name=name,
                tool_call_id=tool_call_id,
            )
            outputs.append(msg)
            TRACE.event(
                "tool.exec.unknown",
                {
                    "run_id": run_id,
                    "attempt": attempt,
                    "tool": name,
                    "tool_call_id": tool_call_id,
                    "args_raw": tool_args_raw,
                },
            )
            continue

        TRACE.event(
            "tool.exec.start",
            {
                "run_id": run_id,
                "attempt": attempt,
                "tool": name,
                "tool_call_id": tool_call_id,
                "args_raw": tool_args_raw,
                "args_norm": tool_args,
            },
        )

        t0 = time.perf_counter()
        ok = True
        try:
            result = tool.invoke(tool_args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
        except Exception as e:
            ok = False
            result = f"Tool error: {type(e).__name__}: {e}"
        dt_ms = int((time.perf_counter() - t0) * 1000)

        TRACE.event(
            "tool.exec.end",
            {
                "run_id": run_id,
                "attempt": attempt,
                "tool": name,
                "tool_call_id": tool_call_id,
                "ok": ok,
                "latency_ms": dt_ms,
                "result_preview": _short(result, 1200),
            },
        )

        outputs.append(
            ToolMessage(
                content=result,
                name=name,
                tool_call_id=tool_call_id,
            )
        )

    TRACE.event(
        "tools.batch.end",
        {
            "run_id": run_id,
            "attempt": attempt,
            "node": "tools",
            "num_tool_messages": len(outputs),
            "tool_messages_preview": [
                {"name": m.name, "content_preview": _short(m.content, 600), "tool_call_id": m.tool_call_id}
                for m in outputs
                if isinstance(m, ToolMessage)
            ],
        },
    )

    return {"messages": outputs}


def critique_node(state: AgentState, config: RunnableConfig):
    """
    Critique using the FULL transcript (system + messages).
    """
    run_id = state.get("run_id", "unknown")
    attempt = int(state.get("attempt", 1))

    transcript_lines = []
    for m in state["messages"]:
        role = m.__class__.__name__
        content = getattr(m, "content", "")
        if isinstance(m, ToolMessage):
            role = f"Tool[{m.name}]"
        transcript_lines.append(f"{role}: {content}")

    transcript = "\n".join(transcript_lines).strip()

    critique_messages = [
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=transcript),
    ]

    TRACE.event(
        "llm.critique.start",
        {
            "run_id": run_id,
            "attempt": attempt,
            "node": "critique",
            "transcript_preview": _short(transcript, 2000),
        },
    )

    t0 = time.perf_counter()
    critique = model.invoke(critique_messages, config=config).content.strip()
    dt_ms = int((time.perf_counter() - t0) * 1000)

    TRACE.event(
        "llm.critique.end",
        {
            "run_id": run_id,
            "attempt": attempt,
            "node": "critique",
            "latency_ms": dt_ms,
            "critique": critique,
        },
    )

    return {"critique": critique}


def reset_for_retry_node(state: AgentState):
    """
    Prepare next attempt:
    - increment attempt counter
    - reset messages to just the original human query
    - store critique as reflection notes
    """
    run_id = state.get("run_id", "unknown")
    next_attempt = int(state.get("attempt", 1)) + 1
    original_query = state["original_query"]
    critique = (state.get("critique") or "").strip()

    TRACE.event(
        "retry.reset",
        {
            "run_id": run_id,
            "from_attempt": int(state.get("attempt", 1)),
            "to_attempt": next_attempt,
            "reflection": critique,
        },
    )

    return {
        "attempt": next_attempt,
        "reflection": critique,
        "messages": [HumanMessage(content=original_query)],
    }


# ----------------------------
# 5) Control flow (edges)
# ----------------------------
def should_continue_tools(state: AgentState):
    """
    If the model requested tool calls, run tools; otherwise go to critique.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    decision = "tools" if tool_calls else "critique"

    TRACE.event(
        "route.after_agent",
        {
            "run_id": state.get("run_id", "unknown"),
            "attempt": int(state.get("attempt", 1)),
            "decision": decision,
            "num_tool_calls": len(tool_calls),
        },
    )
    return decision


def should_retry(state: AgentState):
    """
    If critique is CORRECT. -> END
    else retry if attempts remain
    """
    critique = (state.get("critique") or "").strip()
    attempt = int(state.get("attempt", 1))
    max_attempts = int(state.get("max_attempts", 2))

    if critique == "CORRECT.":
        decision = "end"
    elif attempt >= max_attempts:
        decision = "end"
    else:
        decision = "retry"

    TRACE.event(
        "route.after_critique",
        {
            "run_id": state.get("run_id", "unknown"),
            "attempt": attempt,
            "max_attempts": max_attempts,
            "critique": critique,
            "decision": decision,
        },
    )
    return decision


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("reset_for_retry", reset_for_retry_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue_tools,
        {"tools": "tools", "critique": "critique"},
    )
    workflow.add_edge("tools", "agent")

    workflow.add_conditional_edges(
        "critique",
        should_retry,
        {"retry": "reset_for_retry", "end": END},
    )
    workflow.add_edge("reset_for_retry", "agent")

    return workflow.compile()


# ============================================================
# 6) Run (with optional graph stream debug)
# ============================================================
def main():
    setup_logging(logging.INFO)

    graph = build_graph()

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk net worth' "
        "and summarize in 1 line."
    )

    run_id = uuid.uuid4().hex[:12]
    inputs: AgentState = {
        "run_id": run_id,
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "attempt": 1,
        "max_attempts": 2,
        "reflection": "",
        "critique": "",
    }

    TRACE.event(
        "run.start",
        {"run_id": run_id, "query": query, "tools": [t.name for t in tools]},
    )

    # Option A (recommended for tracing): stream with debug to see node boundaries
    final_state: Optional[AgentState] = None
    for event in graph.stream(inputs, stream_mode="debug"):
        # event is a dict; shape depends on langgraph version.
        # We keep this robust: just log a short snapshot.
        TRACE.event("graph.debug", {"run_id": run_id, "event_preview": _short(event, 2000)})

        # If it's a "values" style event, it may contain the current state
        if isinstance(event, dict) and "values" in event and isinstance(event["values"], dict):
            final_state = event["values"]  # last seen state

    # Fallback: if stream didn't yield final values, do a normal invoke
    if not final_state:
        final_state = graph.invoke(inputs)

    TRACE.event(
        "run.end",
        {
            "run_id": run_id,
            "final_message_preview": _short(final_state["messages"][-1].content, 1200),
            "critique": final_state.get("critique", ""),
        },
    )

    print("\nFinal Answer:\n")
    print(final_state["messages"][-1].content)
    print("\nCritique:\n")
    print(final_state.get("critique", ""))

    print("\nTrace file (JSONL): agent_trace.jsonl")


if __name__ == "__main__":
    main()
