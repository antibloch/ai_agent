import os
import json
import asyncio
import time
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

import re
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel



def _extract_first_json_object(text: str) -> str:
    """Extract the first top-level JSON object from arbitrary text.
    Works even if the model adds <think>, markdown fences, or extra commentary.
    """
    if not text:
        raise ValueError("Empty LLM output")

    # Remove common wrappers (optional)
    cleaned = text.strip()

    # Try to find a fenced json block first
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Otherwise, find the first balanced {...} object
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No '{' found in LLM output")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1].strip()

    raise ValueError("Unbalanced JSON braces in LLM output")








def format_msg(m: BaseMessage) -> str:
    role = m.__class__.__name__
    content = (getattr(m, "content", "") or "").strip()

    # Tool calls (from AIMessage)
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        content += "\n\n[tool_calls]\n" + json.dumps(tool_calls, indent=2)

    # ToolMessage has tool name/id info sometimes
    tool_name = getattr(m, "name", None)
    if tool_name:
        content = f"[tool={tool_name}]\n{content}"

    return f"{role}:\n{content}\n"


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

    history_cursor: int  # index into messages list for incremental printing



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
            model="qwen3.5:cloud", 
            # model="qwen:latest",
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
    model = make_model_chat(temperature=0.0)  # important: reduce drift
    parser = PydanticOutputParser(pydantic_object=Plan)
    tools_description = textual_description_of_tools(tools)

    system = SystemMessage(
        content=(
            "You are a planner. Break the user's request into a short sequence of steps.\n"
            "Rules:\n"
            "- Return ONLY a JSON object. No markdown, no code fences, no <think>.\n"
            "- The JSON must match the schema exactly.\n"
            "- Steps should be concrete, ordered, and cover ALL parts of the request.\n"
            f"{parser.get_format_instructions()}\n"
            "AVAILABLE TOOLS:\n"
            f"{tools_description}"
        )
    )

    exec_system = SystemMessage(
        content=(
            "You are a tool-using execution agent.\n"
            "You will be given ONE plan step at a time.\n"
            "Solve the CURRENT step fully. Use tools as needed.\n"
            "If you did not call a tool, DO NOT claim you did.\n"
        )
    )

    def _parse_plan_with_retries(raw: str, user_query: str, config: RunnableConfig) -> Plan:
        # 1) try direct parse
        try:
            return parser.parse(raw)
        except Exception:
            pass

        # 2) try extracting JSON object and parsing that
        try:
            json_str = _extract_first_json_object(raw)
            return parser.parse(json_str)
        except Exception:
            pass

        # 3) ask model to re-emit ONLY JSON (1 retry)
        repair_prompt = HumanMessage(
            content=(
                "Your previous output was invalid.\n"
                "Re-output ONLY the JSON object that matches the schema. No extra text.\n\n"
                f"USER QUERY:\n{user_query}\n\n"
                "Return ONLY JSON now."
            )
        )
        repaired = model.invoke([system, repair_prompt], config=config).content
        try:
            json_str = _extract_first_json_object(repaired)
            return parser.parse(json_str)
        except Exception as e:
            # 4) last-resort fallback: 1-step plan
            return Plan(steps=[f"Answer the user query directly: {user_query}"])

    def planner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        raw = model.invoke([system, HumanMessage(content=state["user_query"])], config=config).content
        plan_obj = _parse_plan_with_retries(raw, state["user_query"], config)

        return {
            "plan": plan_obj.steps,
            "step_idx": 0,
            "current_step": None,
            "completed": [],
            "history_cursor": 0,  # if you're using step logging
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

    # A small “post-tool” nudge that prevents empty AIMessage after ToolMessage
    POST_TOOL_NUDGE = (
        "Tool result received.\n"
        "Now write the final result for THIS plan step in plain text.\n"
        "- Do NOT call more tools unless strictly necessary.\n"
        "- Be concise and specific.\n"
    )

    def executor_model_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        plan = state.get("plan", [])
        idx = int(state.get("step_idx", 0))
        if idx >= len(plan):
            return {}

        step = plan[idx]
        msgs = list(state.get("messages", []))

        starting_new_step = (state.get("current_step") != step)

        new_messages: List[BaseMessage] = []

        if starting_new_step:
            # Inject the step instruction ONCE per step
            new_messages.append(
                HumanMessage(
                    content=(
                        "PLAN STEP (execute only this step):\n"
                        f"{step}\n\n"
                        "If needed, call tools. Otherwise, provide the final result for this step."
                    )
                )
            )
        else:
            # We are in the tool-loop for the SAME step.
            # If the last message is a ToolMessage, force the model to produce the step result.
            last = msgs[-1] if msgs else None
            if isinstance(last, ToolMessage):
                new_messages.append(HumanMessage(content=POST_TOOL_NUDGE))

        ai = model.invoke(msgs + new_messages, config=config)

        out: Dict[str, Any] = {"messages": [*new_messages, ai]}
        if starting_new_step:
            out["current_step"] = step

        return out

    return executor_model_node


# ----------------------------
# Advance node: record step result and move to next step
# ----------------------------

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

    return f"{role}:\n{content}\n"



def advance_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    idx = int(state.get("step_idx", 0))

    if idx >= len(plan):
        raise RuntimeError("advance_node called but no steps remain.")

    step = plan[idx]
    msgs = list(state.get("messages", []))
    cursor = int(state.get("history_cursor", 0))

    # ----------------------------------------------------
    # 1️⃣ Log intermediate transcript for THIS step
    # ----------------------------------------------------
    print("\n\n============================================================")
    print(f"[STEP {idx}] {step}")
    print("------------------------------------------------------------")

    # for i in range(cursor, len(msgs)):
    #     print(format_message(msgs[i]))

    print("============================================================\n")

    # ----------------------------------------------------
    # 2️⃣ Extract final AI result (robust version)
    # ----------------------------------------------------
    step_result = None
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None)
            content = (m.content or "").strip()
            if not tool_calls and content:
                step_result = content
                break

    if step_result is None:
        step_result = "(No valid final step result was produced.)"

    # ----------------------------------------------------
    # 3️⃣ Update structured state
    # ----------------------------------------------------
    completed = state.get("completed", []) + [(step, step_result)]

    # --- DEBUG: dump full messages list at step end ---
    print("\n[DEBUG] state['messages'] at END of step")
    print(f"Total messages: {len(msgs)}")
    for j, m in enumerate(msgs):
        print(f"\n--- msg[{j}] ---")
        print(format_message(m))


    return {
        "completed": completed,
        "step_idx": idx + 1,
        "current_step": None,
        "history_cursor": len(msgs),  # advance cursor
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
    model = make_model_chat(temperature=0.0)

    FINALIZER_SYSTEM_PROMPT = """\
You are the finalizer.

Task: Write the final user answer by summarizing the executed (step, result) list.

Rules:
- Output only the final answer (no analysis / meta / self-talk; no tool mentions).
- Use only the provided step results; do not add new facts.
- Organize by user sub-questions as short headings with bullet points.
- If any requested part is not supported by the step results, add a final "Missing" section listing what’s missing (1 line each). Otherwise omit "Missing".
- If multiple step results repeat the same info, deduplicate and keep the clearest/latest.
"""

    system = SystemMessage(content=FINALIZER_SYSTEM_PROMPT)

    def finalizer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        completed = state.get("completed", []) or []
        executed = "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in completed])

        msg = HumanMessage(
            content=(
                f"USER QUERY:\n{state.get('user_query','')}\n\n"
                f"EXECUTED STEPS + RESULTS:\n{executed}\n\n"
                "Write the final answer now."
            )
        )

        ai = model.invoke([system, msg], config=config)
        final = (ai.content or "").strip()

        # Retry once if empty (common intermittent behavior in some Ollama setups)
        if not final:
            retry = HumanMessage(
                content="Your last answer was empty. Output the final answer as plain text now."
            )
            ai2 = model.invoke([system, msg, retry], config=config)
            final = (ai2.content or "").strip()

            # Keep the finalizer transcript for debugging if needed
            return {"messages": [msg, ai, retry, ai2], "final_answer": final}

        # Keep the finalizer transcript for debugging if needed
        return {"messages": [msg, ai], "final_answer": final}

    return finalizer_node


# ----------------------------
# Build graph
# ----------------------------
async def build_graph():
    tools = await setup_tools()

    # add hints to tool descriptions to improve performance
    tools = patch_tool_descriptions(tools)

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
    start_time = time.time()
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
        "history_cursor": 0,
    }

    # IMPORTANT: async run so ToolNode uses tool.ainvoke for async-only tools (e.g., MCP)
    out = await graph.ainvoke(state)
    out_msg = out.get("final_answer", "")

    console = Console()
    print("\n\n\nAgent Final Response:")
    print("-----------------------------------------------------------------------")
    console.print(Markdown(out_msg))


    # print("\n\nConversation Transcript:")
    # print("======================================================================")
    # for i, m in enumerate(out.get("messages", [])):
    #     print(f"--- #{i} -------------------------------------------------------------")
    #     print(format_msg(m))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n\n**Total elapsed time**: {elapsed:.2f} seconds")



if __name__ == "__main__":
    asyncio.run(main())
