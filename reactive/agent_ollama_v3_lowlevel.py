import os
import json
from typing import Annotated, Sequence, TypedDict, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ----------------------------
# 1) Graph state
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ----------------------------
# 2) Tools + model
# ----------------------------
def build_tools():
    # Both are "single input" tools
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


tools = build_tools()
tools_by_name = {t.name: t for t in tools}

model = ChatOllama(
    model="qc:latest",
    temperature=0,
    # base_url="http://localhost:11434",  # if you changed host/port
).bind_tools(tools)  # IMPORTANT: enables tool calling


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a tool-using AI assistant.\n"
        "Use PythonREPLTool for any computation.\n"
        "Use DuckDuckGoSearchRun for any web/current info.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct."
    )
)


# ----------------------------
# 3) Nodes
# ----------------------------
def _normalize_tool_args(args: Any):
    """
    Tool-call args sometimes arrive as:
      - dict with a single key (e.g., {"query": "..."} or {"code": "..."})
      - dict with "__arg1" for single-input tools
      - a raw string (less common, but handle)
    We pass through in a way most LC tools accept.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        if "__arg1" in args:
            return args["__arg1"]
        # If it's a 1-key dict, many tools accept the raw value as the single input.
        if len(args) == 1:
            return next(iter(args.values()))
        return args
    return args


def tool_node(state: AgentState):
    """
    Execute tool_calls from the last AI message and return ToolMessage(s).
    """
    last = state["messages"][-1]
    outputs = []

    # Some message types may not have tool_calls; be defensive.
    tool_calls = getattr(last, "tool_calls", None) or []
    for tc in tool_calls:
        name = tc["name"]
        tool = tools_by_name.get(name)
        if tool is None:
            # Return an error ToolMessage so the model can recover.
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
            # ToolMessage.content must be a string
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


def call_model(state: AgentState, config: RunnableConfig):
    """
    Call the model with a system prompt + conversation messages.
    """
    response = model.invoke([SYSTEM_PROMPT] + list(state["messages"]), config=config)
    return {"messages": [response]}


# ----------------------------
# 4) Control flow (edges)
# ----------------------------
def should_continue(state: AgentState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "continue" if tool_calls else "end"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# ----------------------------
# 5) Run
# ----------------------------
def main():
    graph = build_graph()

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk net worth' "
        "and summarize in 1 line."
    )

    inputs = {"messages": [HumanMessage(content=query)]}

    # Option A: single-shot invoke (final state returned)
    out_state = graph.invoke(inputs)
    print(out_state["messages"][-1].content)

    # Option B: stream steps (uncomment to see tool/model turns)
    # for step in graph.stream(inputs, stream_mode="values"):
    #     msg = step["messages"][-1]
    #     msg.pretty_print()


if __name__ == "__main__":
    main()
