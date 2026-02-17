import json
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich import print
import os
import asyncio
from typing import Annotated, Sequence, TypedDict, Any, List

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



MODEL="qc:latest"

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
    memory_kv: str  # renamed from "summary" to avoid priming


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
    return [build_node_stats_tool(), PythonREPLTool()]


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


# ----------------------------
# Token / size helpers (char-based heuristic)
# ----------------------------
def _msg_text_len(m: BaseMessage) -> int:
    c = getattr(m, "content", "")
    if isinstance(c, str):
        return len(c)
    # Some tool messages can be list/dict; count a small constant to avoid zeroing them out.
    return 50


def _total_prompt_chars(system_text: str, memory_text: str, messages: Sequence[BaseMessage]) -> int:
    return len(system_text) + len(memory_text) + sum(_msg_text_len(m) for m in messages)


def _safe_tail(messages: List[BaseMessage], keep_last_n: int) -> List[BaseMessage]:
    """
    Keep the last N messages, but avoid starting with ToolMessage.
    Also ensure the final AIMessage is included in the tail if present.
    """
    if not messages:
        return []

    start = max(0, len(messages) - keep_last_n)

    # Ensure the last AIMessage is in tail (important for keeping final answer)
    last_ai = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai = i
            break
    if last_ai is not None:
        start = min(start, last_ai)

    # Avoid starting with ToolMessage
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1

    return messages[start:]


def _format_for_memory(msgs: Sequence[BaseMessage]) -> str:
    """
    Convert messages into a compact, summarizer-friendly transcript.
    Avoid dumping huge tool payloads: truncate ToolMessage content.
    """
    lines = []
    for m in msgs:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = getattr(m, "content", "")

        if isinstance(m, ToolMessage):
            if isinstance(content, str) and len(content) > 2000:
                content = content[:2000] + "\n...[tool output truncated]..."
            elif not isinstance(content, str):
                # Make tool outputs stable-ish
                content = str(content)[:2000]

        if isinstance(content, str):
            content = content.strip()

        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ----------------------------
# Assistant node
# ----------------------------
def make_assistant_node(tools):
    model = ChatOllama(
        model=MODEL,
        temperature=0,
    ).bind_tools(tools)

    tools_description = textual_description_of_tools(tools)

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
        memory_kv = (state.get("memory_kv") or "").strip()

        sys_msg = SystemMessage(content=BASE_SYSTEM_PROMPT_TEXT)

        # IMPORTANT: memory is a separate SystemMessage and labeled neutrally.
        # No "summary" tokens, no "Summary:" headings.
        mem_msg = None
        if memory_kv:
            mem_msg = SystemMessage(
                content=(
                    "REFERENCE FACTS (internal context):\n"
                    "<facts>\n"
                    f"{memory_kv}\n"
                    "</facts>\n"
                )
            )

        msgs = [sys_msg]
        if mem_msg:
            msgs.append(mem_msg)
        msgs.extend(list(state.get("messages", [])))

        # Debug
        # print("\n--- DEBUG: messages being sent to Ollama ---")
        # for i, m in enumerate(msgs):
        #     c = getattr(m, "content", None)
        #     print(i, type(m).__name__, "content_type=", type(c).__name__, "len=", (len(c) if isinstance(c, str) else "NA"))
        # total_chars = sum(len(getattr(m, "content", "")) for m in msgs if isinstance(getattr(m, "content", ""), str))
        # print("TOTAL_CHARS =", total_chars)
        # print("--- END DEBUG ---\n")

        response = model.invoke(msgs, config=config)
        return {"messages": [response]}

    return assistant_node, BASE_SYSTEM_PROMPT_TEXT


# ----------------------------
# Memory writer node (summarizer)
# ----------------------------
def make_memory_node(base_model: ChatOllama, base_system_prompt: str):
    # Budget is for: base_system_prompt + memory + messages.
    # Char-based heuristic; adjust for your model/context window.
    PROMPT_CHAR_BUDGET = 2000
    KEEP_LAST_N = 1  # keep a bit more; includes final answer + recent context safely

    def memory_node(state: AgentState, config: RunnableConfig):
        messages = list(state.get("messages", []))
        if len(messages) <= KEEP_LAST_N:
            return {}

        memory_kv = (state.get("memory_kv") or "").strip()
        total_chars = _total_prompt_chars(base_system_prompt, memory_kv, messages)

        if total_chars <= PROMPT_CHAR_BUDGET:
            return {}

        tail = _safe_tail(messages, KEEP_LAST_N)
        head = messages[: max(0, len(messages) - len(tail))]
        if not head:
            return {}

        head_text = _format_for_memory(head)

        prompt = [
            SystemMessage(
                content=(
                    "You write internal reference memory for an agent.\n"
                    "Update the existing memory using the new dialogue.\n"
                    "\n"
                    "Output MUST have exactly two sections in this order:\n"
                    "1) FACTS:\n"
                    "   - Each line is 'key: value' (no bullets).\n"
                    "   - Keep stable identifiers.\n"
                    "   - Include numbers, locations, categories, product prices, contact fields if present.\n"
                    "2) NOTES:\n"
                    "   - 1 to 6 short sentences.\n"
                    "   - Capture conclusions, caveats, assumptions, and what the user asked for.\n"
                    "   - Mention missing data (e.g., website URLs not available) if relevant.\n"
                    "\n"
                    "Hard constraints:\n"
                    "- Max total: 22 lines.\n"
                    "- No headings other than exactly 'FACTS:' and 'NOTES:'.\n"
                    "- No markdown, no bullets, no extra sections.\n"
                    "- Do not include tool call IDs or raw JSON blobs.\n"
                    "\n"
                    "Recommended keys (use only what exists):\n"
                    "charities.count\n"
                    "charity.<Name>.location\n"
                    "charity.<Name>.donorCount\n"
                    "charity.<Name>.categories\n"
                    "charity.<Name>.product.<ProductName>\n"
                    "charity.donorCount.mean\n"
                    "charity.donorCount.median\n"
                    "data.missing.websiteUrls (true/false)\n"
                    "\n"
                    "Example:\n"
                    "FACTS:\n"
                    "charities.count: 2\n"
                    "charity.HelpingHands.location: Karachi, Pakistan\n"
                    "charity.HelpingHands.donorCount: 4\n"
                    "NOTES:\n"
                    "User requested all charities, ranking by donors, and mean/median donor counts.\n"
                    "Website URLs had info about state of charities.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"EXISTING FACTS:\n{memory_kv if memory_kv else '(none)'}\n\n"
                    f"NEW DIALOGUE:\n{head_text}"
                )
            ),
        ]

        mem_msg = base_model.invoke(prompt, config=config)
        new_mem = (getattr(mem_msg, "content", "") or "").strip()
        if not new_mem:
            return {}

        return {"memory_kv": new_mem, "messages": tail}

    return memory_node


async def build_graph():
    tools = await setup_tools()

    workflow = StateGraph(AgentState)

    assistant_node, base_system_prompt = make_assistant_node(tools)

    memory_model = ChatOllama(model=MODEL, temperature=0)
    memory_node = make_memory_node(memory_model, base_system_prompt)

    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("memory", memory_node)

    workflow.add_edge(START, "assistant")

    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            END: "memory",
        },
    )

    workflow.add_edge("tools", "assistant")
    workflow.add_edge("memory", END)

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

    inputs = {"messages": [HumanMessage(content=query)], "memory_kv": ""}

    config: RunnableConfig = {"configurable": {"thread_id": "charity-demo-1"}}

    out_state = await graph.ainvoke(inputs, config=config)
    out_msg = out_state["messages"][-1].content

    console = Console()
    print("\n\nAgent Final Response:")
    print("-----------------------------------------------------------------------")
    console.print(Markdown(out_msg))

    memory_kv = (out_state.get("memory_kv") or "").strip()
    if memory_kv:
        print("\n\n[Stored Reference Facts]")
        print("-----------------------------------------------------------------------")
        console.print(Markdown(memory_kv))


if __name__ == "__main__":
    asyncio.run(main())
