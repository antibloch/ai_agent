import os
import json
import asyncio
import requests
from typing import Annotated, Sequence, TypedDict, Any, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich import print

from langchain_core.tools import Tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from langchain_experimental.tools import PythonREPLTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ============================
# 0) Config: summarization trigger
# ============================
MAX_NUM_MSGS = 3          # Trigger summarization when len(messages) > MAX_NUM_MSGS
KEEP_LAST_N = 2            # Keep a safe tail of recent messages (will be adjusted to avoid starting with ToolMessage)
MAX_TOOL_LOOPS = 8        # Max allowed tool loops before forcing summarization to break potential infinite loops.
MODEL = "qc:latest"  # Ollama model for assistant node

# ============================
# 1) Tool description helpers
# ============================

def tool_to_text(t: Any) -> str:
    name = getattr(t, "name", t.__class__.__name__)
    desc = (getattr(t, "description", "") or "").strip()

    schema = None
    args_schema = getattr(t, "args_schema", None)
    if args_schema is not None:
        try:
            if hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
            elif hasattr(args_schema, "schema"):
                schema = args_schema.schema()
        except Exception:
            schema = None

    if schema is None:
        raw_schema = getattr(t, "tool_call_schema", None) or getattr(t, "schema", None)
        if isinstance(raw_schema, dict):
            schema = raw_schema

    parts = [f"- {name}"]
    if desc:
        parts.append(f"  Description: {desc}")
    if schema:
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
    tool_loops: int


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
    return [
        build_node_stats_tool(),
        PythonREPLTool(),
    ]


async def setup_tools():
    local_tools = build_local_tools()

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


# ============================
# 4) Memory summarization helpers
# ============================
def _safe_tail(messages: List[BaseMessage], keep_last_n: int) -> List[BaseMessage]:
    """
    MUST preserve the last user instruction.
    Strategy:
      - Find the last HumanMessage; keep from there to the end.
      - If that slice is smaller than keep_last_n, expand left.
      - Never start with ToolMessage (expand one step left if needed).
    """
    if not messages:
        return []

    # 1) Anchor: last HumanMessage (critical)
    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        # No human in history (rare). Fall back to last N.
        start = max(0, len(messages) - keep_last_n)
    else:
        start = last_human_idx

        # Ensure we keep at least keep_last_n messages (expand left if needed)
        desired_start = max(0, len(messages) - keep_last_n)
        start = min(start, desired_start)

    # 2) Avoid starting with ToolMessage (breaks tool-call pairing)
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1

    return messages[start:]



def _format_for_summary(msgs: Sequence[BaseMessage]) -> str:
    """
    Convert to summarizer-friendly transcript. Truncate large tool outputs.
    """
    lines: List[str] = []
    for m in msgs:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = getattr(m, "content", "")
        if isinstance(m, ToolMessage) and isinstance(content, str) and len(content) > 2000:
            content = content[:2000] + "\n...[truncated tool output]..."
        if isinstance(content, str):
            content = content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ============================
# 5) Assistant node
# ============================
def make_assistant_node(tools):
    model = ChatOllama(model=MODEL, temperature=0).bind_tools(tools)
    tools_description = textual_description_of_tools(tools)

    # NOTE: Keeping tool list in prompt can be huge with MCP tools.
    # If you hit Ollama 500 again, comment out tools_description block.
    BASE_SYSTEM_PROMPT_TEXT = (
        "You are a tool-using AI assistant.\n"
        "\n"
        "AVAILABLE TOOLS:\n"
        f"{tools_description}\n"
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
        "   a) Infer the user's intent (list, compare, rank, compute, etc.).\n"
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
        system_text = BASE_SYSTEM_PROMPT_TEXT
        if summary:
            system_text += (
                "\n\nMEMORY SUMMARY (compressed prior context; treat as long-term memory):\n"
                f"{summary}\n"
            )

        system_msg = SystemMessage(content=system_text)

        # Ensure all message contents are strings for Ollama stability
        msgs: List[BaseMessage] = [system_msg]
        for m in list(state.get("messages", [])):
            c = getattr(m, "content", "")
            if not isinstance(c, str):
                # Coerce structured content to json string
                try:
                    c = json.dumps(c, ensure_ascii=False)
                except Exception:
                    c = str(c)

                if isinstance(m, HumanMessage):
                    m = HumanMessage(content=c)
                elif isinstance(m, SystemMessage):
                    m = SystemMessage(content=c)
                elif isinstance(m, ToolMessage):
                    m = ToolMessage(content=c, tool_call_id=m.tool_call_id)
                else:
                    m = AIMessage(content=c)

            msgs.append(m)

        response = model.invoke(msgs, config=config)
        tool_loops = int(state.get("tool_loops", 0))
        if getattr(response, "tool_calls", None):
            tool_loops += 1

        return {"messages": [response], "tool_loops": tool_loops}

    return assistant_node


# ============================
# 6) Summarizer node (triggered by MAX_NUM_MSGS)
# ============================
def summarize_conversation(state: AgentState, config: RunnableConfig):
    messages = list(state.get("messages", []))
    if len(messages) <= MAX_NUM_MSGS:
        return {}

    existing = (state.get("summary") or "").strip()

    tail = _safe_tail(messages, KEEP_LAST_N)
    head = messages[: max(0, len(messages) - len(tail))]
    if not head:
        return {}

    summarizer_model = ChatOllama(model=MODEl, temperature=0)

    head_text = _format_for_summary(head)

    summarizer_prompt = [
        SystemMessage(content=(
            "You are a memory summarizer for an LLM agent.\n"
            "Update the existing memory summary using the new dialogue.\n"
            "Rules:\n"
            "- Keep durable facts, goals, constraints, decisions.\n"
            "- Keep key tool findings (final numbers, top ranks, important URLs).\n"
            "- Drop chit-chat and duplicates.\n"
            "- Output 6–12 bullet points.\n"
            "- Be concise.\n"
        )),
        HumanMessage(content=f"EXISTING SUMMARY:\n{existing if existing else '(none)'}"),
        HumanMessage(content=f"NEW DIALOGUE:\n{head_text}"),
    ]

    resp = summarizer_model.invoke(summarizer_prompt, config=config)
    new_summary = (resp.content or "").strip()
    if not new_summary:
        return {}

    # Replace entire message history with only tail (safe) and store updated summary
    return {
        "summary": new_summary,
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *tail],
    }


def should_summarize(state: AgentState) -> str:
    # Trigger summarization strictly by MAX_NUM_MSGS
    if len(list(state.get("messages", []))) > MAX_NUM_MSGS:
        return "summarize"
    return "assistant"



def route_after_assistant(state: AgentState) -> str:
    msgs = list(state.get("messages", []))
    last = msgs[-1] if msgs else None

    # If model wants tools
    if last is not None and getattr(last, "tool_calls", None):
        if int(state.get("tool_loops", 0)) >= MAX_TOOL_LOOPS:
            return "summarize"   # break loop safely
        return "tools"

    # No tool calls: only summarize if threshold exceeded
    if len(msgs) > MAX_NUM_MSGS:
        return "summarize"

    # Otherwise end this run
    return END


# ============================
# 7) Build graph (correct wiring!)
# ============================
async def build_graph():
    tools = await setup_tools()

    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", make_assistant_node(tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("summarize", summarize_conversation)

    workflow.add_edge(START, "assistant")

    # # assistant -> tools OR summarize (end-of-loop) -> (maybe assistant again)
    # workflow.add_conditional_edges(
    #     "assistant",
    #     tools_condition,
    #     {"tools": "tools", END: "summarize"},
    # )

    # After assistant responds, we check if it wants to call tools. If yes, route to tools node; if no, check if we need to summarize (based on MAX_NUM_MSGS). If summarization is triggered, we route to summarize node; if not, we loop back to assistant for the next turn. This allows for multi-turn interactions without summarization if the conversation is still short.
    workflow.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {"tools": "tools", "summarize": "summarize"},
    )


    # CRITICAL: tools must go back to assistant so it can interpret tool outputs
    workflow.add_edge("tools", "assistant")

    # After summarization, go back to assistant (next tick) if you want multi-turn with same invoke,
    # but for invoke/ainvoke it's fine to end here. We'll just end.
    workflow.add_edge("summarize", END)

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================
# 8) Run
# ============================
async def main():
    graph = await build_graph()

    query = (
        "Find me all available charities, "
        "Which charities have the highest donor count, "
        "What are the mean and median of donor counts across charities, please calculate using python if needed, "
        "Provide info about charities from their websites, "
        "What kind of items do they provide, and what are the price descriptions for these items"
    )

    # NOTE: agent state uses `summary`
    inputs = {"messages": [HumanMessage(content=query)], "summary": ""}

    # IMPORTANT: thread_id enables checkpointed memory across turns
    config: RunnableConfig = {"configurable": {"thread_id": "charity-demo-1"}}

    out_state = await graph.ainvoke(inputs, config=config)
    out_msg = out_state["messages"][-1].content

    console = Console()
    print("\n\nAgent Final Response:")
    print("-----------------------------------------------------------------------")
    console.print(Markdown(out_msg))

    # Optional: show stored memory summary if summarization triggered
    summary = (out_state.get("summary") or "").strip()
    if summary:
        print("\n\nMemory Summary Stored:")
        print("-----------------------------------------------------------------------")
        console.print(Markdown(summary))


if __name__ == "__main__":
    asyncio.run(main())
