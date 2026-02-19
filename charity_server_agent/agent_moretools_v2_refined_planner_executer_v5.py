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
    messages: Annotated[Sequence[BaseMessage], add_messages]
    prompt_messages: List[BaseMessage]   # NEW (no reducer, so overwrites)
    user_query: str
    plan: List[str]
    step_idx: int
    current_step: Optional[str]
    completed: List[Tuple[str, str]]
    final_answer: str
    history_cursor: int
    context_summaries: List[str]
    step_start_idx: int


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



def make_summarizer_chat(choice: str = "ollama") -> BaseChatModel:
    # Separate model instance; no tools bound; temperature 0 for stability
    return make_model_chat(temperature=0.0, bind_tools=None, choice=choice)



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
            "history_cursor": 0,
            "context_summaries": [],
            "step_start_idx": 1,  # base is now [exec_system] only
            "messages": [
                exec_system,
            ],
            "prompt_messages": [
                exec_system,
            ],
        }


    return planner_node



# ----------------------------
# Executor model node
# ----------------------------
def make_executor_model_node(tools):
    model = make_model_chat(temperature=0.3, bind_tools=tools)

    POST_TOOL_NUDGE = (
        "Tool result received.\n"
        "Now write the final result for THIS plan step in plain text.\n"
        "- Do NOT call more tools unless strictly necessary.\n"
        "- Be concise and specific.\n"
    )

    EMPTY_OUTPUT_NUDGE = (
        "Your last message was empty.\n"
        "You MUST now produce the final answer for this plan step in plain text.\n"
        "Do not call tools unless absolutely required.\n"
        "Do not output an empty message."
    )

    # How many consecutive empty assistant replies to tolerate within ONE node call
    MAX_EMPTY_RETRIES = int(os.getenv("MAX_EMPTY_RETRIES", "4"))

    def _is_empty_final(ai: AIMessage) -> bool:
        # Empty if: no tool calls AND no text content
        has_tools = bool(getattr(ai, "tool_calls", None))
        content = (getattr(ai, "content", "") or "").strip()
        return (not has_tools) and (not content)

    def executor_model_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        plan = state.get("plan", [])
        idx = int(state.get("step_idx", 0))
        if idx >= len(plan):
            return {}

        step = plan[idx]

        msgs_full: List[BaseMessage] = list(state.get("messages", []))          # authoritative log
        msgs_prompt: List[BaseMessage] = list(state.get("prompt_messages", [])) # System/(Context)

        starting_new_step = (state.get("current_step") != step)

        out: Dict[str, Any] = {}
        new_messages: List[BaseMessage] = []

        # ----------------------------
        # Build model_input
        # ----------------------------
        if starting_new_step:
            # Mark where THIS step begins in the full transcript (before we append PLAN STEP)
            out["current_step"] = step
            out["step_start_idx"] = len(msgs_full)

            plan_step_msg = HumanMessage(
                content=(
                    "PLAN STEP (execute only this step):\n"
                    f"{step}\n\n"
                    "If needed, call tools. Otherwise, you MUST produce a clear final answer in plain text. "
                    "Do not output an empty message."
                )
            )
            new_messages.append(plan_step_msg)

            # Executor input (no user query): System + (Context if any) + Plan Step
            model_input: List[BaseMessage] = msgs_prompt + [plan_step_msg]

        else:
            # Continuing same step: include ONLY this step's trace (not full history)
            step_start_idx = int(state.get("step_start_idx", 0))
            current_step_msgs = msgs_full[step_start_idx:]  # PLAN STEP + tool loop + ToolMessages + etc.

            last_full = msgs_full[-1] if msgs_full else None
            if isinstance(last_full, ToolMessage):
                new_messages.append(HumanMessage(content=POST_TOOL_NUDGE))
            elif (
                isinstance(last_full, AIMessage)
                and not (last_full.content or "").strip()
                and not getattr(last_full, "tool_calls", None)
            ):
                new_messages.append(HumanMessage(content=EMPTY_OUTPUT_NUDGE))

            model_input = msgs_prompt + current_step_msgs + new_messages

        # ----------------------------
        # Invoke model with bounded retries for empty final outputs
        # ----------------------------
        ai = model.invoke(model_input, config=config)

        # We'll append everything we generated in THIS node call
        appended: List[BaseMessage] = [*new_messages, ai]

        # If the model returned empty *and* it wasn't a tool call, retry a few times
        # within this same node invocation (prevents graph-level spam loops).
        retries = 0
        while _is_empty_final(ai) and retries < MAX_EMPTY_RETRIES:
            retries += 1
            retry_msg = HumanMessage(content=EMPTY_OUTPUT_NUDGE)

            # Keep the retry context tight: the same model_input plus the retry instruction.
            ai = model.invoke(model_input + [retry_msg], config=config)

            appended.extend([retry_msg, ai])

            # If the model chooses to call tools on retry, stop here and let ToolNode run.
            if getattr(ai, "tool_calls", None):
                break

        out["messages"] = appended
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






def _safe_truncate(s: str, n: int = 4000) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "\n…(truncated)"


def summarize_step_transcript_llm(
    summarizer: BaseChatModel,
    step: str,
    step_msgs: List[BaseMessage],
    config: RunnableConfig,
) -> str:
    """
    Summarizes ONLY the step trace (PLAN STEP + tool calls + tool outputs + intermediate chatter).
    It intentionally does NOT summarize the final natural-language result.
    """
    lines: List[str] = []
    for m in step_msgs:
        role = m.__class__.__name__
        content = (getattr(m, "content", "") or "").strip()

        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None)
            if tc:
                content += "\n[tool_calls]\n" + json.dumps(tc, indent=2)

        if isinstance(m, ToolMessage):
            tool_name = getattr(m, "name", None)
            if tool_name:
                content = f"[tool={tool_name}]\n{content}"

        lines.append(f"{role}:\n{content}" if content else f"{role}:(empty)")

    transcript = _safe_truncate("\n\n".join(lines), 6000)

    system = SystemMessage(
        content=(
            "You are a step-trace summarizer for a planner-executor agent.\n"
            "Summarize ONLY what happened in this one step's TRACE.\n"
            "The final natural-language result will be attached separately, so DO NOT restate it.\n"
            "\n"
            "Rules:\n"
            "- Do NOT mention internal frameworks (e.g., LangGraph, ToolNode).\n"
            "- Preserve entity names and numeric values exactly as seen.\n"
            "- Focus on tool outputs and key facts derived from them.\n"
            "- Keep it compact and structured.\n"
            "- Output plain text ONLY.\n"
            "\n"
            "Format:\n"
            "STEP: <one line>\n"
            "TRACE SUMMARY:\n"
            "- <key action/fact>\n"
            "- <key action/fact>\n"
        )
    )

    user = HumanMessage(
        content=(
            f"STEP:\n{step}\n\n"
            f"STEP TRACE (to summarize):\n{transcript}\n\n"
            "Write the trace summary now."
        )
    )

    try:
        ai = summarizer.invoke([system, user], config=config)
        summary = (ai.content or "").strip()
        if summary:
            return summary
    except Exception as e:
        if os.getenv("DEBUG_MESSAGES", "0") == "1":
            print(f"[summarizer error] {e}")

    # Safe fallback
    return (
        f"STEP: {step}\n"
        f"TRACE SUMMARY:\n"
        f"- (trace summary unavailable)\n"
    )




def make_advance_node(summarizer: BaseChatModel):
    def advance_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:

        plan = state.get("plan", [])
        idx = int(state.get("step_idx", 0))
        if idx >= len(plan):
            raise RuntimeError("advance_node called but no steps remain.")

        step = plan[idx]
        msgs_full = list(state.get("messages", []))          # full transcript
        msgs_prompt = list(state.get("prompt_messages", [])) # compact prompt base

        # 1) Extract final natural-language AI result for THIS step from FULL transcript
        step_start_idx = int(state.get("step_start_idx", 1))
        step_msgs = msgs_full[step_start_idx:]

        step_result = None

        # Only inspect THIS step's messages
        for m in reversed(step_msgs):
            if isinstance(m, AIMessage):
                content = (m.content or "").strip()
                tool_calls = getattr(m, "tool_calls", None)

                # Must be a final natural-language result
                if content and not tool_calls:
                    step_result = content
                    break

        if not step_result:
            step_result = "(No valid final step result was produced.)"


        completed = state.get("completed", []) + [(step, step_result)]


        # DEBUG: print exactly what you want
        if os.getenv("DEBUG_MESSAGES", "0") == "1":
            print("\n\n============================================================")
            print(f"STEP-[{idx}] {step}")
            print("------------------------------------------------------------")

            # Order: System, User, Context (from prompt_messages) + step transcript (from full messages)
            combined = list(msgs_prompt) + list(step_msgs)

            for j, m in enumerate(combined):
                print(f"\n--- msg[{j}] ---")
                print(format_message(m))

            print("\n============================================================\n")

        # 3) Build trace messages to summarize (exclude final natural-language result)
        trace_msgs = list(step_msgs)
        for k in range(len(trace_msgs) - 1, -1, -1):
            m = trace_msgs[k]
            if isinstance(m, AIMessage):
                tool_calls = getattr(m, "tool_calls", None)
                content = (m.content or "").strip()
                if not tool_calls and content:
                    trace_msgs.pop(k)
                    break

        # 4) Summarize ONLY the trace
        trace_summary = summarize_step_transcript_llm(
            summarizer=summarizer,
            step=step,
            step_msgs=trace_msgs,
            config=config,
        )

        # 5) Append verbatim final result into the context entry
        context_entry = (
            f"{trace_summary.strip()}\n\n"
            f"FINAL RESULT (verbatim):\n{step_result.strip()}"
        )

        context_summaries = list(state.get("context_summaries", []))
        context_summaries.append(context_entry)

        MAX_CONTEXT_STEPS = 20
        if len(context_summaries) > MAX_CONTEXT_STEPS:
            context_summaries = context_summaries[-MAX_CONTEXT_STEPS:]

        # 6) Rebuild prompt_messages (this overwrites cleanly)
        # prompt_messages always: [System, User Query, Context]
        base_exec_system = msgs_prompt[0] if len(msgs_prompt) >= 1 else msgs_full[0]

        context_block = "\n\n---\n\n".join(context_summaries)
        context_msg = HumanMessage(content="CONTEXT (previous step summaries):\n\n" + context_block)

        new_prompt_msgs = [base_exec_system, context_msg]


        return {
            "completed": completed,
            "context_summaries": context_summaries,
            "prompt_messages": new_prompt_msgs,  # IMPORTANT: executor uses this next
            "step_idx": idx + 1,
            "current_step": None,
            "step_start_idx": len(msgs_full),    # next step transcript starts at end of full list
        }

    return advance_node




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




def should_continue(state: AgentState) -> str:
    plan = state.get("plan", [])
    idx = int(state.get("step_idx", 0))
    return "executor" if idx < len(plan) else "finalize"



def executor_routing(state: AgentState) -> str:
    last = (list(state.get("messages", [])) or [None])[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "advance"



# ----------------------------
# Build graph
# ----------------------------
async def build_graph():
    tools = await setup_tools()

    # add hints to tool descriptions to improve performance
    tools = patch_tool_descriptions(tools)

    summarizer_model = make_summarizer_chat(choice="ollama")  # separate instance

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", make_planner_node(tools))
    workflow.add_node("executor_model", make_executor_model_node(tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("advance", make_advance_node(summarizer_model))  # <- changed
    workflow.add_node("finalizer", make_finalizer_node())

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor_model")

    # If tool_calls exist -> tools, else -> advance
    workflow.add_conditional_edges(
        "executor_model",
        executor_routing,
        {"tools": "tools", "advance": "advance", "executor_model": "executor_model"},
    )



    # tools execute and append ToolMessages, then go back to model for the SAME step
    workflow.add_edge("tools", "executor_model")

    # after advancing, either do next step or finalize
    workflow.add_conditional_edges(
        "advance",
        should_continue,
        {"executor": "executor_model", "finalize": "finalizer"},
    )


    workflow.add_edge("finalizer", END)

    return workflow.compile()


# ----------------------------
# Run
# ----------------------------
async def main():
    print("DEBUG_MESSAGES =", os.getenv("DEBUG_MESSAGES"))

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
        "context_summaries": [],
        "step_start_idx": 0,
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
