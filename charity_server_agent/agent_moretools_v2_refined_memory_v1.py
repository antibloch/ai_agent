import json
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich import print
import os
import asyncio
from typing import Annotated, Sequence, TypedDict

from langchain_core.tools import Tool

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver  # Added MemorySaver
from langchain_core.messages import RemoveMessage  # Added RemoveMessage

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




class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str  # Added summary field




def build_node_stats_tool():
    BASE_URL = "http://localhost:3000"

    # Canonical tool names (must match Node TOOL_REGISTRY keys exactly)
    CANONICAL_TOOLS = [
        "charity_donor_count",
        "charity_impactlife",
        "charity_donor_amount",
        "charity_total_donation",
        "charity_items_category",
        "charity_product_price_description",
        "charity_blogs",
        "charity_address",
        "charity_country_availability",
        "charity_contact_info",  # NOTE: typo-canonical in your Node router
    ]

    def call_node_stats(tool_name: str) -> str:
        """
        Calls Node.js GET /api/stats?q=<tool_name>
        Returns JSON string.
        """
        tool_name = (tool_name or "").strip()

        if not tool_name:
            return json.dumps({
                "ok": False,
                "error": "Tool name is required (empty input).",
                "valid_tools": CANONICAL_TOOLS,
            })

        # Strict contract enforcement (no aliases allowed)
        if tool_name not in CANONICAL_TOOLS:
            return json.dumps({
                "ok": False,
                "error": "Invalid tool name for Node stats endpoint.",
                "provided": tool_name,
                "valid_tools": CANONICAL_TOOLS,
            })

        try:
            r = requests.get(
                f"{BASE_URL}/api/stats",
                params={"q": tool_name},
                timeout=5,
            )
            r.raise_for_status()
            return json.dumps(r.json())
        except requests.RequestException as e:
            return json.dumps({
                "ok": False,
                "error": str(e),
                "tool": tool_name,
            })

    return Tool(
        name="get_charity_stats",
        description=(
            "Fetch internal charity data from Node-js server.\n"
            "Input MUST be EXACTLY one tool name.\n"
            "The input must be EXACTLY one of the following canonical tool names:\n"
            "1. charity_donor_count: Number of unique donors per charity.\n"
            "2. charity_impactlife: Human impact/lives touched metrics.\n"
            "3. charity_donor_amount: Total currency amount donated.\n"
            "4. charity_total_donation: Breakdown of product-specific donation counts.\n"
            "5. charity_items_category: Categories of aid provided (e.g., Food, Health).\n"
            "6. charity_product_price_description: Details on specific charity products/vouchers.\n"
            "7. charity_blogs: Narrative updates and blog posts from the charities.\n"
            "8. charity_address: Physical locations and HQ details (Good for listing charities).\n"
            "9. charity_country_availability: Where these charities operate.\n"
            "10. charity_contact_info: Emails, phones, and websites (Use this exact spelling).\n"
            "\nReturns JSON: {ok, tool, query, data, meta}.\n"
            "'data' field contains the actual response from the Node server for the given tool query."
        ),
        func=call_node_stats,
    )




def build_local_tools():
    # Both are "single input" tools
    return [
            build_node_stats_tool(), 
            # DuckDuckGoSearchRun(), 
            PythonREPLTool()
            ]




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
    return [*local_tools, *mcp_tools]




def make_assistant_node(tools):
    """
    We keep your model+prompt behavior, but bind the FULL tool set (local + MCP).
    """
    model = ChatOllama(
        model="qc:latest",
        temperature=0,
        # base_url="http://localhost:11434",
    ).bind_tools(tools)

    tools_description = textual_description_of_tools(tools)


    SYSTEM_PROMPT_TEMPLATE = (
        "You are a tool-using AI assistant.\n"
        "\n"
        "AVAILABLE TOOLS:\n{tools_description}\n"
        "\n"
        "GENERAL OPERATING RULES:\n"
        "1) If the user request is short, ambiguous, or underspecified, DO NOT refuse.\n"
        "   First rewrite the request internally into a clearer, measurable objective.\n"
        "   Then proceed.\n"
        "\n"
        "2) Every part of the user's request MUST be addressed.\n"
        "   If multiple steps are required, perform them sequentially.\n"
        "\n"
        "3) All factual claims about charities MUST be grounded in get_charity_stats tool output.\n"
        "   If you did not call a tool, DO NOT claim you did.\n"
        "\n"
        "4) Tool routing policy:\n"
        "   a) Infer the user's intent (list, compare, rank, summarize, compute, etc.).\n"
        "   b) If the question concerns charity information, ALWAYS call get_charity_stats.\n"
        "   c) If uncertain which canonical tool name to use, choose the most general overview tool\n"
        "      (e.g., charity_address for listing charities).\n"
        "   d) Use the MINIMUM number of tool calls required.\n"
        "\n"
        "5) Clarification policy:\n"
        "   If the request cannot be uniquely resolved, ask ONE precise clarification question,\n"
        "   but ALSO provide a best-effort answer using the most relevant available tool.\n"
        "\n"
        "6) Output policy:\n"
        "   Provide a direct answer to EACH part of the query.\n"
        "   Be concise, structured, and factual.\n"
        "\n"
        "Summary of conversation so far:\n{summary}"
    )

    def assistant_node(state: AgentState, config: RunnableConfig):
        summary = state.get("summary", "")
        system_message = SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(
                tools_description=tools_description, summary=summary
            )
        )
        response = model.invoke([system_message] + list(state["messages"]), config=config)
        return {"messages": [response]}

    return assistant_node


def summarize_conversation(state: AgentState):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization model
    model = ChatOllama(model="qc:latest", temperature=0)

    messages = state["messages"]

    # We want to preserve the Initial User Prompt (messages[0])
    # And the last 2 messages (messages[-2:])
    # We summarize everything in betweeen (messages[1:-2])
    
    if len(messages) <= 3:
        return {}

    to_summarize = messages[1:-2]

    # Generate a summary of the conversation so far
    if summary:
        
        # If a summary already exists, we use a different prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # We invoke model with the messages to summarize + instruction
    # Note: we construct a temp list for invoking the summarizer
    summarizer_input = to_summarize + [HumanMessage(content=summary_message)]
    response = model.invoke(summarizer_input)
    
    # We delete the messages we just summarized
    delete_messages = [RemoveMessage(id=m.id) for m in to_summarize]
    
    return {"summary": response.content, "messages": delete_messages}


def should_summarize(state: AgentState):
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than 6 messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    return END



async def build_graph():
    tools = await setup_tools()

    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", make_assistant_node(tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("summarize_conversation", summarize_conversation)

    workflow.add_edge(START, "assistant")

    # Route assistant -> tools if tool_calls exist, else END
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {"tools": "tools", END: END},
    )

    workflow.add_conditional_edges(
        "tools",
        should_summarize,
        {"summarize_conversation": "summarize_conversation", END: END},
    )
    
    workflow.add_edge("summarize_conversation", "assistant")
    
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)





# ----------------------------
# 7) Run
# ----------------------------
async def main():
    graph = await build_graph()


    # First turn
    config = {"configurable": {"thread_id": "1"}}
    
    query = (
        " Find me all available charities, "
        " Which charities have the highest donor count "
        # " What are the mean and median of donor counts across charities, please calculate using python if needed, "
        # " Provide info about charities from their websites, "
        # " What kind of items do they provide, and what are the price descriptions for these items"
        # " and summarize all this in 1 line."
    )
    
    print("\n\n--- TURN 1 ---")
    inputs = {"messages": [HumanMessage(content=query)]}
    async for event in graph.astream(inputs, config=config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # Second turn
    print("\n\n--- TURN 2 ---")
    query2 = "What did I just ask you about? Also, calculate the mean of donor counts."
    inputs2 = {"messages": [HumanMessage(content=query2)]}
    async for event in graph.astream(inputs2, config=config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # Option B: stream steps (uncomment to see tool/model turns)
    # for step in graph.stream(inputs, stream_mode="values"):
    #     msg = step["messages"][-1]
    #     msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
