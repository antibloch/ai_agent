import json
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich import print
import os
import asyncio
from typing import Annotated, Sequence, TypedDict, Any, List, Optional

from langchain_core.tools import Tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from langchain_experimental.tools import PythonREPLTool
from langchain_mcp_adapters.client import MultiServerMCPClient



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




class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str




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



# ----------------------------
# Memory summarization helpers
# ----------------------------
def _count_chars(messages: Sequence[BaseMessage]) -> int:
    total = 0
    for m in messages:
        c = getattr(m, "content", "")
        if isinstance(c, str):
            total += len(c)
    return total


def _safe_tail(messages: List[BaseMessage], keep_last_n: int) -> List[BaseMessage]:
    """
    Make a 'safe' tail so we don't start with a ToolMessage (which can break tool-call pairing).
    Simple heuristic:
      - take last N
      - if it starts with ToolMessage, expand left until it doesn't (or we hit start)
      - prefer starting at a HumanMessage if possible
    """
    if not messages:
        return []

    start = max(0, len(messages) - keep_last_n)

    # Prefer starting at a HumanMessage within the tail window (more stable for ReAct)
    for i in range(start, len(messages)):
        if isinstance(messages[i], HumanMessage):
            start = i
            break

    # Avoid starting with ToolMessage
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1

    return messages[start:]


def _format_for_summary(msgs: Sequence[BaseMessage]) -> str:
    """
    Convert messages into a compact, summarizer-friendly transcript.
    Avoid dumping huge tool payloads: truncate ToolMessage content.
    """
    lines = []
    for m in msgs:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = getattr(m, "content", "")

        if isinstance(m, ToolMessage) and isinstance(content, str):
            # Tool outputs can be huge—truncate hard.
            if len(content) > 2000:
                content = content[:2000] + "\n...[truncated tool output]..."

        if isinstance(content, str):
            content = content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

# ----------------------------
# Memory summarization helpers
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

    tools_description = textual_description_of_tools(tools)


    BASE_SYSTEM_PROMPT_TEXT = (
                            "You are a tool-using AI assistant.\n"
                            "\n"
                            "AVAILABLE TOOLS:\n" + tools_description +
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
                            )



    def assistant_node(state: AgentState, config: RunnableConfig):
        summary = (state.get("summary") or "").strip()
        summary_msg = ""
        if summary:
            summary_msg = (
                "MEMORY SUMMARY (compressed prior context; treat as long-term memory):\n"
                f"{summary}\n"
            )

        # Build final system prompt text (string) then wrap in SystemMessage
        system_text = BASE_SYSTEM_PROMPT_TEXT + ("\n\n" + summary_msg if summary_msg else "")
        SYSTEM_PROMPT = SystemMessage(content=system_text)

        # response = model.invoke([SYSTEM_PROMPT] + list(state["messages"]), config=config)
        def _debug_messages(msgs):
            print("\n--- DEBUG: messages being sent to Ollama ---")
            for i, m in enumerate(msgs):
                c = getattr(m, "content", None)
                print(i, type(m).__name__, "content_type=", type(c).__name__, "len=", (len(c) if isinstance(c, str) else "NA"))
            total_chars = sum(len(getattr(m, "content", "")) for m in msgs if isinstance(getattr(m, "content", ""), str))
            print("TOTAL_CHARS =", total_chars)
            print("--- END DEBUG ---\n")

        msgs = [SYSTEM_PROMPT] + list(state["messages"])
        _debug_messages(msgs)
        response = model.invoke(msgs, config=config)

        return {"messages": [response]}

    return assistant_node, BASE_SYSTEM_PROMPT_TEXT



# ----------------------------
#  Summarizer node
# ----------------------------
def make_summarize_node(base_model: ChatOllama, base_system_prompt: str):
    # Tune:
    MAX_HISTORY_CHARS = 1400  # summarize when exceeded
    KEEP_LAST_N = 2          # keep recent messages verbatim (safe tail will adjust)

    def summarize_node(state: AgentState, config: RunnableConfig):
        messages = list(state.get("messages", []))
        if len(messages) <= KEEP_LAST_N:
            return {}

        if _count_chars(messages) < MAX_HISTORY_CHARS:
            return {}

        tail = _safe_tail(messages, KEEP_LAST_N)
        head = messages[: max(0, len(messages) - len(tail))]

        if not head:
            return {}

        existing_summary = (state.get("summary") or "").strip()
        head_text = _format_for_summary(head)

        summarizer_prompt = [
            SystemMessage(content=(
                "You are a memory summarizer for an LLM agent.\n"
                "Update the existing memory summary using the new dialogue.\n"
                "Rules:\n"
                "- Keep durable facts only (names, preferences, goals, constraints, decisions).\n"
                "- Keep key tool findings (final numeric results, best-ranked items, important URLs).\n"
                "- Drop chatter and duplicates.\n"
                "- Output 6–12 bullet points max.\n"
                "- Be concise.\n"
            )),
            HumanMessage(content=(
                f"EXISTING SUMMARY:\n{existing_summary if existing_summary else '(none)'}\n\n"
                f"NEW DIALOGUE TO INCORPORATE:\n{head_text}"
            )),
        ]

        summary_msg = base_model.invoke(summarizer_prompt, config=config)
        new_summary = (getattr(summary_msg, "content", "") or "").strip()
        if not new_summary:
            return {}

        # IMPORTANT: we do NOT keep the old head messages; we keep only tail + updated summary
        return {"summary": new_summary, "messages": tail}

    return summarize_node


def should_summarize(state: AgentState) -> str:
    # Only summarize after a full tool loop finishes (i.e., after assistant returns without tool calls),
    # OR you can summarize more aggressively. Here: always check after assistant.
    # We route to "summarize" unconditionally; summarize_node itself is a no-op unless thresholds exceeded.
    return "summarize"

# ----------------------------
#  Summarizer node
# ----------------------------



async def build_graph():
    tools = await setup_tools()

    workflow = StateGraph(AgentState)

    assistant_node, base_system_prompt = make_assistant_node(tools)

    # base model for summarizer (can be same as assistant, but not bound to tools)
    summarizer_model = ChatOllama(model="qc:latest", temperature=0)
    summarize_node = make_summarize_node(summarizer_model, base_system_prompt)

    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "assistant")

    # assistant -> tools or summarize (and eventually end)
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            END: "summarize",  # no tool calls => possibly summarize, then end
        },
    )

    workflow.add_edge("tools", "assistant")
    workflow.add_edge("summarize", END)

    # Add checkpointing so summary/messages persist by thread_id
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)





async def main():
    graph = await build_graph()

    query = (
        "Find me all available charities, "
        "Which charities have the highest donor count, "
        "What are the mean and median of donor counts across charities, please calculate using python if needed, "
        "Provide info about charities from their websites, "
        "What kind of items do they provide, and what are the price descriptions for these items"
    )

    inputs = {"messages": [HumanMessage(content=query)], "summary": ""}

    # IMPORTANT: provide thread_id so memory persists across multiple invokes
    config: RunnableConfig = {"configurable": {"thread_id": "charity-demo-1"}}

    out_state = await graph.ainvoke(inputs, config=config)
    out_msg = out_state["messages"][-1].content

    console = Console()
    print("\n\nAgent Final Response:")
    print("-----------------------------------------------------------------------")
    console.print(Markdown(out_msg))

    # Show compressed memory (if summarization triggered)
    summary = (out_state.get("summary") or "").strip()
    if summary:
        print("\n\n[Memory Summary Stored]")
        print("-----------------------------------------------------------------------")
        console.print(Markdown(summary))


if __name__ == "__main__":
    asyncio.run(main())
