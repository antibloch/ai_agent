import os
import json
from typing import Annotated, Sequence, TypedDict, Any, Optional

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
import re

_CORRECT_RE = re.compile(r"^\s*CORRECT\s*[\.\!\u2705✅]*\s*$", re.IGNORECASE)

def is_critique_correct(critique: str) -> bool:
    return bool(_CORRECT_RE.match(critique or ""))


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


# ----------------------------
# 4) Nodes
# ----------------------------
def call_model(state: AgentState, config: RunnableConfig):
    """
    Tool-capable model call for the current attempt.
    """
    system_msg = _system_message_for_attempt(state)
    response = model.invoke([system_msg] + list(state["messages"]), config=config)
    return {"messages": [response]}


def tool_node(state: AgentState):
    """
    Execute tool_calls from the last AI message and return ToolMessage(s).
    """
    last = state["messages"][-1]
    outputs = []

    tool_calls = getattr(last, "tool_calls", None) or []
    for tc in tool_calls:
        name = tc["name"]
        tool = tools_by_name.get(name)

        if tool is None:
            outputs.append(
                ToolMessage(
                    content=f"Unknown tool: {name}",
                    name=name,
                    tool_call_id=tc.get("id", ""),
                )
            )
            continue

        tool_args = _normalize_tool_args(tc.get("args"))
        try:
            result = tool.invoke(tool_args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
        except Exception as e:
            result = f"Tool error: {type(e).__name__}: {e}"

        outputs.append(
            ToolMessage(
                content=result,
                name=name,
                tool_call_id=tc.get("id", ""),
            )
        )

    return {"messages": outputs}


def critique_node(state: AgentState, config: RunnableConfig):
    """
    Critique using the FULL transcript (system + messages).
    """
    # Build a readable transcript for critique (include tool outputs too)
    # We DO NOT rely on hidden traces; we critique the explicit message list.
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
    critique = model.invoke(critique_messages, config=config).content.strip()
    return {"critique": critique}


def reset_for_retry_node(state: AgentState):
    """
    Prepare next attempt:
    - increment attempt counter
    - reset messages to just the original human query
    - store critique as reflection notes
    """
    next_attempt = int(state.get("attempt", 1)) + 1
    original_query = state["original_query"]
    critique = (state.get("critique") or "").strip()

    return {
        "attempt": next_attempt,
        "reflection": critique,
        # Reset transcript for the next attempt (avoids compounding confusion)
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
    return "tools" if tool_calls else "critique"


def should_retry(state: AgentState):
    """
    If critique is CORRECT. -> END
    else retry if attempts remain
    """
    critique = (state.get("critique") or "").strip()
    attempt = int(state.get("attempt", 1))
    max_attempts = int(state.get("max_attempts", 2))

    # if critique == "CORRECT.":
    #     return "end"

    if is_critique_correct(critique):
        return "end"

    if attempt >= max_attempts:
        return "end"

    return "retry"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("reset_for_retry", reset_for_retry_node)

    workflow.set_entry_point("agent")

    # Tool loop: agent -> (tools or critique)
    workflow.add_conditional_edges(
        "agent",
        should_continue_tools,
        {"tools": "tools", "critique": "critique"},
    )
    workflow.add_edge("tools", "agent")

    # Critique -> (retry or end)
    workflow.add_conditional_edges(
        "critique",
        should_retry,
        {"retry": "reset_for_retry", "end": END},
    )
    workflow.add_edge("reset_for_retry", "agent")

    return workflow.compile()


# ----------------------------
# 6) Run
# ----------------------------
def main():
    graph = build_graph()

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk net worth' "
        "and summarize in 1 line."
    )

    inputs: AgentState = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "attempt": 1,
        "max_attempts": 2,
        "reflection": "",
        "critique": "",
    }

    out_state = graph.invoke(inputs)
    print("\nFinal Answer:\n")
    print(out_state["messages"][-1].content)
    print("\nCritique:\n")
    print(out_state.get("critique", ""))

    # Optional: stream turns for debugging
    # for step in graph.stream(inputs, stream_mode="values"):
    #     msg = step["messages"][-1]
    #     print("\n--- step ---")
    #     msg.pretty_print()


if __name__ == "__main__":
    main()
