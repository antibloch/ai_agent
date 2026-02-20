import os
import json
import asyncio
import requests
import time
from typing import Annotated, Sequence, TypedDict, Any, Dict, List, Tuple, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich import print

from pydantic import BaseModel, Field

from langchain_core.tools import Tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_experimental.tools import PythonREPLTool

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


from langchain_mcp_adapters.client import MultiServerMCPClient

import re
from langchain_core.exceptions import OutputParserException

from langchain_core.language_models.chat_models import BaseChatModel



def make_model_chat(temperature: float, bind_tools: Optional[list] = None, choice: str="ollama") -> BaseChatModel:
    if choice == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        chat = ChatOpenAI(
            # model="nvidia/nemotron-3-nano-30b-a3b:free",
            model = "qwen/qwen3-coder:free",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
        if bind_tools:
            chat = chat.bind_tools(bind_tools)

    elif choice == "ollama": 
        chat = ChatOllama(
            model = "qc:latest",
            # model="qwen3.5:cloud", 
            # model = "qwen:latest",
            temperature = temperature
            )
        if bind_tools:
            chat = chat.bind_tools(bind_tools)

    else:
        raise ValueError(f"Invalid model choice: {choice}")
    return chat




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
    system_prompt: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    empty_tool_retries: int



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





HINTS = {
    "Python_REPL": (
        "Only printed output is returned.\n"
        "ALWAYS end your code with print(...) of the final result.\n"
        "Prefer minimal Python; avoid heavy libraries unless necessary.\n"
        "If computing statistics, print a compact JSON-like dict."
    ),
    "get_charity_stats": (
        "Input must be EXACTLY one valid tool name string.\n"
        "The response is JSON. The actual results are in the 'data' field.\n"
        "Do not guess tool names — use only listed valid names."
    ),
    "fetch_url": (
        "Use this when you already have a specific URL and need the page content.\n"
        "Prefer this over web search when the task says 'read this URL' or 'extract details from this page'."
    ),
    "fetch_urls": (
        "Fetch multiple URLs in one call when you need to read several pages quickly."
    ),
}


def patch_tool_descriptions(tools: list) -> list:
    """
    Docstring for patch_tool_descriptions with additional hints for better performance
    
    :param tools: Description
    :type tools: list
    :return: Description
    :rtype: list
    """
    patched_tools = []

    for tool in tools:
        name = getattr(tool, "name", tool.__class__.__name__)
        original_desc = (getattr(tool, "description", "") or "").strip()

        hint = HINTS.get(name)

        if hint:
            new_desc = (
                original_desc
                + "\n\nUSAGE HINTS:\n"
                + hint
            )

            try:
                tool.description = new_desc
            except Exception:
                # Some tools may not allow mutation (StructuredTool edge cases)
                pass

        patched_tools.append(tool)

    return patched_tools



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




def get_last_final_ai_text(messages: Sequence[BaseMessage]) -> str:
    """
    Returns the last AI message content that is not a tool-call instruction.
    (In many runs, the last AIMessage is already the final. This is a bit safer.)
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                continue
            return (m.content or "").strip()
    # fallback
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return (m.content or "").strip()
    return ""


def build_system_prompt_with_history(base_system_prompt: str, history: List[Tuple[str, str]]) -> str:
    """
    Inject ALL prior (user_query, final_answer) pairs into the system prompt.
    No full transcript is provided.
    """
    if not history:
        return base_system_prompt

    blocks = []
    for i, (q, a) in enumerate(history, start=1):
        q = (q or "").strip()
        a = (a or "").strip()
        blocks.append(
            f"\n---\n"
            f"Turn #{i}\n"
            f"Previous user query:\n{q}\n\n"
            f"Your previous final answer:\n{a}\n"
        )

    injection = (
        "\n\n"
        "PAST TURN CONTEXT (for continuity; not a full transcript):\n"
        + "".join(blocks) +
        "\n---\n"
        "Use this context to stay consistent. Do NOT mention this block unless the user asks.\n"
    )
    return base_system_prompt + injection


def make_assistant_node(tools):
    model = make_model_chat(temperature=0.0, bind_tools=tools)
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
        sys_text = state.get("system_prompt") or BASE_SYSTEM_PROMPT_TEXT
        system_msg = SystemMessage(content=sys_text)
        response = model.invoke([system_msg] + list(state["messages"]), config=config)
        return {"messages": [response]}

    return assistant_node, BASE_SYSTEM_PROMPT_TEXT, model



def make_retry_assistant_node(model, base_system_prompt_text: str, max_retries: int = 2):
    def retry_node(state: AgentState, config: RunnableConfig):
        retries = int(state.get("empty_tool_retries", 0))
        if retries >= max_retries:
            # Stop retrying; let normal flow continue/end
            return {}

        sys_text = state.get("system_prompt") or base_system_prompt_text
        system_msg = SystemMessage(content=sys_text)

        # Re-ask the model given the same context; model should decide what to do
        response = model.invoke([system_msg] + list(state["messages"]), config=config)

        return {
            "empty_tool_retries": retries + 1,
            "messages": [response],
        }
    return retry_node


MAX_EMPTY_TOOL_RETRIES = 2

def make_retry_empty_tool_node(model, base_system_prompt_text: str):
    def retry_empty_tool_node(state: AgentState, config: RunnableConfig):
        retries = int(state.get("empty_tool_retries", 0))
        if retries >= MAX_EMPTY_TOOL_RETRIES:
            print(f"[RETRY] Skipping: reached MAX_EMPTY_TOOL_RETRIES={MAX_EMPTY_TOOL_RETRIES}")
            return {}

        msgs = list(state.get("messages", []))
        last = msgs[-1] if msgs else None

        # Extra safety: only retry if last message is truly an empty ToolMessage
        if not (isinstance(last, ToolMessage) and len((last.content or "").strip()) == 0):
            return {}

        print(f"[RETRY] Detected EMPTY ToolMessage. Retrying assistant (attempt {retries+1}/{MAX_EMPTY_TOOL_RETRIES})")

        # Strong nudge so it re-runs Python tool and PRINTS the final result
        retry_hint = HumanMessage(
            content=(
                "The previous tool returned EMPTY output. "
                "If you call Python_REPL again, you MUST end with print(...) "
                "so the tool returns visible output. Re-run the necessary tool call."
            )
        )

        sys_text = state.get("system_prompt") or base_system_prompt_text
        system_msg = SystemMessage(content=sys_text)

        response = model.invoke([system_msg] + msgs + [retry_hint], config=config)

        return {
            "empty_tool_retries": retries + 1,
            "messages": [response],
        }

    return retry_empty_tool_node


async def build_graph():
    tools = await setup_tools()
    tools = patch_tool_descriptions(tools)

    assistant_node, base_system_prompt_text, model = make_assistant_node(tools)
    retry_node = make_retry_empty_tool_node(model, base_system_prompt_text)

    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("retry_empty_tool", retry_node)

    workflow.add_edge(START, "assistant")

    # assistant -> tools if tool_calls else END
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {"tools": "tools", END: END},
    )

    # tools -> retry if last ToolMessage empty, else assistant
    def route_after_tools(state: AgentState) -> str:
        if last_tool_was_empty(state):
            return "retry_empty_tool"
        return "assistant"

    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"retry_empty_tool": "retry_empty_tool", "assistant": "assistant"},
    )

    # retry -> tools if tool_calls else END (or assistant; tools_condition will choose END when no tool_calls)
    workflow.add_conditional_edges(
        "retry_empty_tool",
        tools_condition,
        {"tools": "tools", END: END},
    )

    graph = workflow.compile()
    return graph, base_system_prompt_text




def is_empty_tool_message(m: BaseMessage) -> bool:
    if not isinstance(m, ToolMessage):
        return False
    content = (getattr(m, "content", "") or "")
    return len(content.strip()) == 0

def last_tool_was_empty(state: AgentState) -> bool:
    msgs = list(state.get("messages", []))
    if not msgs:
        return False
    return is_empty_tool_message(msgs[-1])


def format_message(m: BaseMessage) -> str:
    role = m.__class__.__name__
    content = (getattr(m, "content", "") or "").strip()

    if isinstance(m, AIMessage):
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            content += "\n\n[tool_calls]\n" + json.dumps(tool_calls, indent=2)

    if isinstance(m, ToolMessage):
        tool_name = getattr(m, "name", None)
        if tool_name:
            content = f"[tool={tool_name}]\n{content}"

    if isinstance(m, ToolMessage):
        tool_name = getattr(m, "name", None)
        c = (getattr(m, "content", "") or "")
        if tool_name:
            content = f"[tool={tool_name}]\n" + (c if c.strip() else "<EMPTY TOOL OUTPUT>")

    return f"{role}:\n{content}\n"


def print_system_prompt_for_run(run_label: str, system_prompt_text: str):
    console = Console()
    print("\n--- SYSTEM PROMPT " + run_label + " ---\n")
    console.print(Markdown(system_prompt_text))
    print("\n--------------------------\n")


# ----------------------------
# 7) Run
# ----------------------------
async def run_one_query(graph, system_prompt_text: str, query: str, run_label: str = "") -> Tuple[List[BaseMessage], str]:
    # Print prompt ONCE per run
    print_system_prompt_for_run(run_label or "", system_prompt_text)

    inputs = {
        "system_prompt": system_prompt_text,
        "empty_tool_retries": 0,
        "messages": [HumanMessage(content=query)],
    }

    cursor = 0
    last_msgs: List[BaseMessage] = []

    async for step in graph.astream(inputs, stream_mode="values"):
        msgs = list(step.get("messages", []))
        last_msgs = msgs

        for m in msgs[cursor:]:
            print(format_message(m))

        cursor = len(msgs)

    final_text = get_last_final_ai_text(last_msgs)
    return last_msgs, final_text



def clip(text: str, max_chars: int = 1500) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[clipped]..."





async def main():
    start_time = time.time()
    graph, base_system_prompt_text = await build_graph()

    # You control how many runs and what queries:
    queries = [
        (
            "Find me all available charities, "
            "Which charities have the highest donor count, "
            "What are the mean and median of donor counts across charities, please calculate using python if needed."
        ),
        "Now summarize what has been done so far.",
        # "Third query here...",
        # "Fourth query here...",
    ]

    history: List[Tuple[str, str]] = []  # [(user_query, final_answer), ...]

    console = Console()

    for run_idx, query in enumerate(queries, start=1):
        run_label = f"RUN #{run_idx}"

        # For run #1, system prompt is base only.
        # For run #k (k>1), inject ALL previous turns.
        system_prompt_this_run = build_system_prompt_with_history(
            base_system_prompt_text,
            history=history,  # all prior runs only
        )

        print("\n====================")
        print(run_label)
        print("====================\n")

        msgs, final = await run_one_query(
            graph,
            system_prompt_this_run,
            query,
            run_label=run_label
        )

        print(f"\nAgent Final Response ({run_label}):\n-----------------------")
        console.print(Markdown(final))

        # Append THIS run's (query, final answer) so it appears in the next run's prompt
        history.append((query, final))
        # history.append((clip(query, 800), clip(final, 2000)))    # if want to clip

    elapsed = time.time() - start_time
    print(f"\n\n**Total elapsed time**: {elapsed:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())