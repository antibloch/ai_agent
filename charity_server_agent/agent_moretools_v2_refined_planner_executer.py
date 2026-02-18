import os
import json
import asyncio
import requests
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


api_key = os.getenv("OPENROUTER_API_KEY")


# ----------------------------
# Tool text helper (optional, for prompt clarity)
# ----------------------------
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


# ----------------------------
# Plan schema
# ----------------------------
class Plan(BaseModel):
    steps: List[str] = Field(
        description="A short ordered list of concrete steps to solve the user's request."
    )


# ----------------------------
# Graph state
# ----------------------------
class AgentState(TypedDict, total=False):
    # Full running transcript used by ToolNode + model
    messages: Annotated[Sequence[BaseMessage], add_messages]

    user_query: str
    plan: List[str]
    step_idx: int

    # NEW: track which step is currently being executed to avoid re-injecting it every tool loop
    current_step: Optional[str]

    completed: List[Tuple[str, str]]  # (step, result)
    final_answer: str


# ----------------------------
# Node stats tool (your tool)
# ----------------------------
def build_node_stats_tool():
    BASE_URL = "http://localhost:3000"

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
        "charity_contact_info",
    ]

    def call_node_stats(tool_name: str) -> str:
        tool_name = (tool_name or "").strip()
        if not tool_name:
            return json.dumps(
                {"ok": False, "error": "Tool name is required.", "valid_tools": CANONICAL_TOOLS}
            )

        if tool_name not in CANONICAL_TOOLS:
            return json.dumps(
                {
                    "ok": False,
                    "error": "Invalid tool name for Node stats endpoint.",
                    "provided": tool_name,
                    "valid_tools": CANONICAL_TOOLS,
                }
            )

        try:
            r = requests.get(f"{BASE_URL}/api/stats", params={"q": tool_name}, timeout=10)
            r.raise_for_status()
            return json.dumps(r.json())
        except requests.RequestException as e:
            return json.dumps({"ok": False, "error": str(e), "tool": tool_name})

    return Tool(
        name="get_charity_stats",
        description=(
            "Fetch internal charity data from Node-js server.\n"
            "Input MUST be EXACTLY one tool name.\n"
            "Valid tool names:\n"
            + "\n".join([f"- {t}" for t in CANONICAL_TOOLS])
            + "\nReturns JSON: {ok, tool, query, data, meta}.\n"
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

    # MCP tools
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
# LLM helper
# ----------------------------
def make_model_chat(temperature: float, bind_tools: Optional[list] = None, choice: str="openrouter") -> ChatOpenAI:
    if choice == "openrouter":
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
            model="qwen3.5:cloud", 
            temperature=temperature
            )
        if bind_tools:
            chat = chat.bind_tools(bind_tools)

    else:
        raise ValueError(f"Invalid model choice: {choice}")
    return chat


# ----------------------------
# Planner node
# ----------------------------
def make_planner_node(tools):
    model = make_model_chat(temperature=0.3)
    parser = PydanticOutputParser(pydantic_object=Plan)

    tools_description = textual_description_of_tools(tools)

    system = SystemMessage(
        content=(
            "You are a planner. Break the user's request into a short sequence of steps.\n"
            "Rules:\n"
            "- Output ONLY valid JSON that matches the schema.\n"
            "- Steps should be actionable and reference tools if relevant, but do NOT run tools.\n"
            "- CRITICAL: cover EVERY distinct user sub-request explicitly. If the user asks 3 things, the plan must include steps for all 3.\n"
            "- If the user asks to 'find all available charities', include a step that explicitly enumerates them.\n"
            f"{parser.get_format_instructions()}\n"
            "AVAILABLE TOOLS:\n"
            f"{tools_description}"
        )
    )


    # This is the stable executor system message (kept in transcript)
    exec_system = SystemMessage(
        content=(
            "You are a tool-using execution agent.\n"
            "You will be given ONE plan step at a time.\n"
            "Goal: solve the CURRENT step fully, using tools as needed.\n"
            "When a tool returns results, incorporate them and either call another tool or provide the step's final result.\n"
            "If you did not call a tool, DO NOT claim you did.\n"
            "Keep step results concise and factual.\n"
        )
    )

    def planner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        raw = model.invoke([system, HumanMessage(content=state["user_query"])], config=config).content
        plan_obj = parser.parse(raw)

        # IMPORTANT: include the original user query in the executor transcript ONCE
        # so the executor always has global context without repeating it per tool loop.
        return {
            "plan": plan_obj.steps,
            "step_idx": 0,
            "current_step": None,
            "completed": [],
            "messages": [
                exec_system,
                HumanMessage(content=f"USER QUERY (global context):\n{state['user_query']}"),
            ],
        }

    return planner_node


# ----------------------------
# Executor model node
# ----------------------------
def make_executor_model_node(tools):
    model = make_model_chat(temperature=0.3, bind_tools=tools)

    def executor_model_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        plan = state.get("plan", [])
        idx = int(state.get("step_idx", 0))

        if idx >= len(plan):
            return {}

        step = plan[idx]
        msgs = list(state.get("messages", []))

        # MAIN FLOW FIX:
        # Inject the step instruction ONCE per step (not on every tool loop).
        starting_new_step = (state.get("current_step") != step)

        new_messages: List[BaseMessage] = []
        if starting_new_step:
            new_messages.append(
                HumanMessage(
                    content=(
                        "PLAN STEP (execute only this step):\n"
                        f"{step}\n\n"
                        "If needed, call tools. Otherwise, provide the final result for this step."
                    )
                )
            )

        # If continuing after tools, do NOT repeat the step instruction.
        ai = model.invoke(msgs + new_messages, config=config)

        out: Dict[str, Any] = {"messages": [*new_messages, ai]}
        if starting_new_step:
            out["current_step"] = step
        return out

    return executor_model_node


# ----------------------------
# Advance node: record step result and move to next step
# ----------------------------
def advance_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    idx = int(state.get("step_idx", 0))
    if idx >= len(plan):
        return {}

    step = plan[idx]

    # SECONDARY/ROBUSTNESS FIX:
    # Grab the most recent AIMessage that has NO tool calls (the step's final result).
    step_result = None
    for m in reversed(list(state.get("messages", []))):
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None)
            if not tool_calls:
                step_result = m.content
                break

    if step_result is None:
        step_result = "(No final step result was produced.)"

    completed = state.get("completed", []) + [(step, step_result)]
    return {
        "completed": completed,
        "step_idx": idx + 1,
        "current_step": None,  # reset so next step gets injected once
    }


def should_continue(state: AgentState) -> str:
    plan = state.get("plan", [])
    idx = int(state.get("step_idx", 0))
    return "executor_model" if idx < len(plan) else "finalizer"


def executor_routing(state: AgentState) -> str:
    msgs = list(state.get("messages", []))
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "advance"


# ----------------------------
# Finalizer node
# ----------------------------
def make_finalizer_node():
    model = make_model_chat(temperature=0.2)

    system = SystemMessage(
        content=(
            "You are a finalizer.\n"
            "Produce the final answer to the user.\n"
            "Use ONLY the executed step results provided.\n"
            "If something is missing/insufficient, say so and explain what is missing.\n"
            "Keep it concise.\n"
        )
    )

    def finalizer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        executed = "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in state.get("completed", [])])
        msg = HumanMessage(
            content=(
                f"USER QUERY:\n{state['user_query']}\n\n"
                f"EXECUTED STEPS + RESULTS:\n{executed}\n\n"
                "FINAL ANSWER:"
            )
        )
        final = model.invoke([system, msg], config=config).content
        return {"final_answer": final}

    return finalizer_node


# ----------------------------
# Build graph
# ----------------------------
async def build_graph():
    tools = await setup_tools()

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", make_planner_node(tools))
    workflow.add_node("executor_model", make_executor_model_node(tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("advance", advance_node)
    workflow.add_node("finalizer", make_finalizer_node())

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor_model")

    # If tool_calls exist -> tools, else -> advance
    workflow.add_conditional_edges(
        "executor_model",
        executor_routing,
        {"tools": "tools", "advance": "advance"},
    )

    # tools execute and append ToolMessages, then go back to model for the SAME step
    workflow.add_edge("tools", "executor_model")

    # after advancing, either do next step or finalize
    workflow.add_conditional_edges(
        "advance",
        should_continue,
        {"executor_model": "executor_model", "finalizer": "finalizer"},
    )

    workflow.add_edge("finalizer", END)

    return workflow.compile()


# ----------------------------
# Run
# ----------------------------
async def main():
    graph = await build_graph()

    query = (
        "Find me all available charities. "
        " Which charities have the highest donor count, "
        " What are the mean and median of donor counts across charities, please calculate using python if needed, "
    )

    state: AgentState = {
        "user_query": query,
        "plan": [],
        "step_idx": 0,
        "current_step": None,
        "completed": [],
        "final_answer": "",
        "messages": [],
    }

    # IMPORTANT: async run so ToolNode uses tool.ainvoke for async-only tools (e.g., MCP)
    out = await graph.ainvoke(state)
    out_msg = out.get("final_answer", "")

    console = Console()
    print("\n\nAgent Final Response:")
    print("-----------------------------------------------------------------------")
    console.print(Markdown(out_msg))


if __name__ == "__main__":
    asyncio.run(main())
