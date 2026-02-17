import json
import requests
from langchain_core.tools import Tool

import os
from typing import Annotated, Sequence, TypedDict, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

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
# 0) Define Tools
# ----------------------------
# def build_node_stats_tool():
#     BASE_URL = "http://localhost:3000"

#     # Canonical tool names (the ones returned in payload.tool)
#     CANONICAL_TOOLS = [
#         "charity_donor_count",
#         "charity_impactlife",
#         "charity_donor_amount",
#         "charity_total_donation",
#         "charity_items_category",
#         "charity_product_price_description",
#         "charity_blogs",
#         "charity_address",
#         "charity_country_availability",
#         "chairty_contact_info",  # NOTE: typo-canonical in your toolMap
#     ]

#     # Aliases supported by your Node router (handleToolQuery.toolMap)
#     # We normalize these to canonical names for stricter behavior.
#     ALIASES = {
#         "donor_count": "charity_donor_count",
#         "donors_count": "charity_donor_count",

#         "impactlife": "charity_impactlife",
#         "impact_life": "charity_impactlife",

#         "donor_amount": "charity_donor_amount",
#         "donation_amount": "charity_donor_amount",

#         "total_donation": "charity_total_donation",
#         "product_total_donation": "charity_total_donation",

#         "items_category": "charity_items_category",
#         "product_categories": "charity_items_category",

#         "product_price_description": "charity_product_price_description",
#         "products_info": "charity_product_price_description",

#         "blogs": "charity_blogs",

#         "address": "charity_address",

#         "country_availability": "charity_country_availability",

#         "charity_contact_info": "chairty_contact_info",  # both map to the same handler
#         "contact_info": "chairty_contact_info",
#     }

#     # The full set we allow as input to THIS Python tool
#     VALID_INPUTS = sorted(set(CANONICAL_TOOLS) | set(ALIASES.keys()))

#     def call_node_stats(tool_name: str) -> str:
#         """
#         Calls Node.js GET /api/stats?q=<tool_or_alias>
#         Returns JSON string.
#         """
#         raw = (tool_name or "").strip()
#         if not raw:
#             return json.dumps({
#                 "ok": False,
#                 "error": "Tool name is required (empty input).",
#                 "valid_tools": CANONICAL_TOOLS,
#                 "valid_aliases": sorted(ALIASES.keys()),
#             })

#         # Normalize alias -> canonical (recommended)
#         normalized = ALIASES.get(raw.lower(), raw)

#         # Hard guard: keeps agent on-contract
#         if normalized not in CANONICAL_TOOLS and raw.lower() not in ALIASES:
#             return json.dumps({
#                 "ok": False,
#                 "error": "Invalid tool name for Node stats endpoint.",
#                 "provided": raw,
#                 "normalized": normalized,
#                 "valid_inputs": VALID_INPUTS,
#             })

#         # IMPORTANT:
#         # Your Node server accepts q as either canonical or alias.
#         # We'll send the normalized canonical to keep things consistent.
#         try:
#             r = requests.get(
#                 f"{BASE_URL}/api/stats",
#                 params={"q": normalized},
#                 timeout=5,
#             )
#             r.raise_for_status()
#             return json.dumps(r.json())
#         except requests.RequestException as e:
#             return json.dumps({"ok": False, "error": str(e), "tool": normalized})

#     # return Tool(
#     #     name="get_charity_stats",
#     #     description=(
#     #         "Fetch internal charity data from the Node.js service.\n"
#     #         "Input MUST be exactly one tool name or alias (no natural language).\n"
#     #         f"Canonical tools: {', '.join(CANONICAL_TOOLS)}.\n"
#     #         f"Aliases accepted: {', '.join(sorted(ALIASES.keys()))}.\n"
#     #         "Returns JSON envelope: {ok, tool, query, data, meta}."
#     #     ),
#     #     func=call_node_stats,
#     # )

#     return Tool(
#     name="get_charity_stats",
#     description=(
#         "Fetch internal charity data from Node-js server.\n"
#         "Input MUST be EXACTLY one tool name.\n"
#         "The input must be EXACTLY one of the following canonical tool names:\n"
#         f"1. charity_donor_count: Number of unique donors per charity.\n"
#         f"2. charity_impactlife: Human impact/lives touched metrics.\n"
#         f"3. charity_donor_amount: Total currency amount donated.\n"
#         f"4. charity_total_donation: Breakdown of product-specific donation counts.\n"
#         f"5. charity_items_category: Categories of aid provided (e.g., Food, Health).\n"
#         f"6. charity_product_price_description: Details on specific charity products/vouchers.\n"
#         f"7. charity_blogs: Narrative updates and blog posts from the charities.\n"
#         f"8. charity_address: Physical locations and HQ details (Good for listing charities).\n"
#         f"9. charity_country_availability: Where these charities operate.\n"
#         f"10. chairty_contact_info: Emails, phones, and websites (Note: Use this specific spelling).\n"
#         "Returns JSON: {ok, tool, query, data, meta}."
#     ),
#     func=call_node_stats,
#     )

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
        "chairty_contact_info",  # NOTE: typo-canonical in your Node router
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
            "10. chairty_contact_info: Emails, phones, and websites (Use this exact spelling).\n"
            "\nReturns JSON: {ok, tool, query, data, meta}.\n"
            "'data' field contains the actual response from the Node server for the given tool query."
        ),
        func=call_node_stats,
    )




# ----------------------------
# 1) Build Tools
# ----------------------------
def build_tools():
    # Both are "single input" tools
    return [
            build_node_stats_tool(), 
            # DuckDuckGoSearchRun(), 
            PythonREPLTool()
            ]

tools = build_tools()
tools_by_name = {t.name: t for t in tools}

tools_description = textual_description_of_tools(tools)



# ----------------------------
# 2) Graph state
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# --------------------------------------------
# 3) Defining model as base of LLM agent
# --------------------------------------------
model = ChatOllama(
    model="qc:latest",
    temperature=0,
    # base_url="http://localhost:11434",  # if you changed host/port
).bind_tools(tools)  # IMPORTANT: enables tool calling



# ----------------------------
# 4) Defining system prompt
# ----------------------------
# SYSTEM_PROMPT = SystemMessage(
#     content=(
#         "You are a tool-using AI assistant.\n"
#         "Use build_node_stats_tool for any info related to charities.\n"
#         "If you did not call a tool, DO NOT claim you did.\n"
#         "Be concise and correct."
#     )
# )
# SYSTEM_PROMPT = SystemMessage(
#     content=(
#         "You are a tool-using AI assistant.\n"
#         "\n"
#         "General operating rules:\n"
#         "1) If the user request is short, ambiguous, or underspecified, DO NOT refuse.\n"
#         "   First, rewrite the user request into a more explicit version that makes the goal measurable.\n"
#         "   Then proceed.\n"
#         "2) Make sure all parts of the user request are addressed, even if they seem to require multiple steps or tools.\n"
#         "3) Always ground factual claims in tool outputs when tools are available.\n"
#         "   If you did not call a tool, DO NOT claim you did.\n"
#         "4) Tool routing policy (robust to ambiguity):\n"
#         "   a) Infer the user's intent (e.g., list, compare, rank, summarize, fetch details).\n"
#         "   b) Select the SINGLE most relevant tool call that is likely to return a superset of needed data.\n"
#         "      Prefer broad/overview tools over narrow ones when uncertain.\n"
#         "   c) If multiple tools are needed, call the minimum number.\n"
#         "   d) If tool inputs must be chosen from a fixed set (tool-name router), choose the closest matching\n"
#         "      canonical tool name; if uncertain, choose the most general 'listing/overview' tool.\n"
#         "5) Clarification policy:\n"
#         "   If the request cannot be uniquely resolved, ask ONE targeted clarification question.\n"
#         "   But ALSO provide the best-effort answer using the most general tool available.\n"
#         "6) Output policy:\n"
#         "   Provide a direct answer to EACH and EVERY part of user query.\n"
#         "   Be concise.\n"
#     )
# )

SYSTEM_PROMPT = SystemMessage(
    content=(
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
)





# ----------------------------
# 5) Nodes
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
# 6) Control flow (edges)
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
# 7) Run
# ----------------------------
def main():
    graph = build_graph()


    query = (
        "Find me all available charities, "
        " Which charities have the highest donor count, "
        " What are the mean and median of donor counts across charities, please calculate using python if needed. "
        # "and summarize all this in 1 line."
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
