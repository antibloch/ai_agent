import os
import asyncio
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

from langchain_mcp_adapters.client import MultiServerMCPClient

import json
from typing import Any

def tool_to_text(t: Any) -> str:
    name = getattr(t, "name", t.__class__.__name__)
    desc = (getattr(t, "description", "") or "").strip()

    # Try to expose arg schema (best-effort)
    schema = None
    args_schema = getattr(t, "args_schema", None)
    if args_schema is not None:
        try:
            # Pydantic v1/v2 compat-ish
            if hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
            elif hasattr(args_schema, "schema"):
                schema = args_schema.schema()
        except Exception:
            schema = None

    # Some tools may have raw schema differently
    if schema is None:
        raw_schema = getattr(t, "tool_call_schema", None) or getattr(t, "schema", None)
        if isinstance(raw_schema, dict):
            schema = raw_schema

    parts = [f"- {name}"]
    if desc:
        parts.append(f"  Description: {desc}")
    if schema:
        # keep compact; don’t dump huge schemas
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if props:
            parts.append(f"  Args: {', '.join(props.keys())}")
        if required:
            parts.append(f"  Required: {', '.join(required)}")
    return "\n".join(parts)

def textual_description_of_tools(tools) -> str:
    return "\n\n".join(tool_to_text(t) for t in tools)


# ----------------------------
# 1) Graph state (same)
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ----------------------------
# 2) Tools (same local tools)
# ----------------------------
def build_local_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


# ----------------------------
# 3) System prompt (same)
# ----------------------------
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
# 4) MCP tools setup (reference-style)
# ----------------------------

#-----------------------------------------------------------------------


MCP_TOOL_HINTS = {
    "fetch_url": (
        "Use this when you already have a specific URL and need the page content. "
        "Prefer this over web search when the task says 'read this URL' or 'extract details from this page'."
    ),
    "fetch_urls": (
        "Fetch multiple URLs in one call when you need to read several pages quickly."
    ),
}


def patch_tool_descriptions(tools):
    for t in tools:
        name = getattr(t, "name", "")
        if name in MCP_TOOL_HINTS:
            base = (getattr(t, "description", "") or "").strip()
            hint = MCP_TOOL_HINTS[name].strip()
            # Append your hint; don’t overwrite server-provided detail
            t.description = (base + "\n\n" + "Hint: " + hint).strip()
    return tools

#----------------------------------------------------------------------------


async def setup_tools():
    """
    Adds an MCP server that can fetch a URL's content.
    Uses stdio transport, launched as a subprocess (like the reference code).
    """
    local_tools = build_local_tools()

    # "fetcher-mcp" is an MCP server that exposes fetch_url / fetch_urls (via npx).
    # Source: MCP server listing describing fetch_url tool and how to run it.
    # If you prefer a different server, swap command/args accordingly.
    client = MultiServerMCPClient(
        {
            "fetch": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "fetcher-mcp"],
            }
        }
    )

    mcp_tools = await client.get_tools()
    #----------------------------------------------------
    
    mcp_tools = patch_tool_descriptions(mcp_tools)

    for t in mcp_tools:
        print("NAME:", getattr(t, "name", None))
        print("DESC:", (getattr(t, "description", "") or "")[:200], "\n---")

    #----------------------------------------------------
    return [*local_tools, *mcp_tools]


# ----------------------------
# 5) Assistant node (reference-style)
# ----------------------------
def make_assistant_node(tools):
    """
    We keep your model+prompt behavior, but bind the FULL tool set (local + MCP).
    """
    model = ChatOllama(
        model="qc:latest",
        temperature=0,
        # base_url="http://localhost:11434",
    ).bind_tools(tools)

    tool_text = textual_description_of_tools(tools)

    #----------------------------------------------------
    print("tools_text:\n", tool_text)  # Debug: see the tool descriptions being fed into the prompt
    asnd
    #----------------------------------------------------

    AUG_SYSTEM_PROMPT = SystemMessage(
        content=SYSTEM_PROMPT.content + "\n\nAvailable tools:\n" + tool_text
    )

    def assistant_node(state: AgentState, config: RunnableConfig):
        response = model.invoke([AUG_SYSTEM_PROMPT] + list(state["messages"]), config=config)
        return {"messages": [response]}

    return assistant_node


# ----------------------------
# 6) Build graph (reference-style)
# ----------------------------
async def build_graph():
    tools = await setup_tools()

    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", make_assistant_node(tools))
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "assistant")

    # Route assistant -> tools if tool_calls exist, else END
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {"tools": "tools", END: END},
    )

    workflow.add_edge("tools", "assistant")
    return workflow.compile()


# ----------------------------
# 7) Run (same user prompt)
# ----------------------------
async def main():
    graph = await build_graph()

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then find out about 'Elon Musk's residence' "
        "and then find me info at 'https://github.com/antibloch', "
        "and summarize all this in 1 line."
    )

    inputs = {"messages": [HumanMessage(content=query)]}

    out_state = await graph.ainvoke(inputs)
    print(out_state["messages"][-1].content)

    # Optional: stream
    # async for step in graph.astream(inputs, stream_mode="values"):
    #     msg = step["messages"][-1]
    #     msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
