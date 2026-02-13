import os
import json
from typing import Annotated, Sequence, TypedDict, Any, Optional, List, Dict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ----------------------------
# 0) Tunables (explicit control knobs)
# ----------------------------
MAX_STEPS_DEFAULT = 12          # hard stop
MEMORY_EVERY = 6                # summarize every N steps (tune)
MAX_TOOL_RETRIES = 2            # simple reliability
REPEAT_STUCK_WINDOW = 3         # detect repeating tool patterns


# ----------------------------
# 1) Graph state
# ----------------------------
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Explicit control / reliability
    step: int
    max_steps: int

    # Lightweight memory (summarized)
    memory: str

    # Debug / stopping heuristics
    last_tool_signatures: List[str]  # rolling window of tool call "signatures"
    stop_reason: str


# ----------------------------
# 2) Tools + model
# ----------------------------
def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


tools = build_tools()
tools_by_name = {t.name: t for t in tools}

model = ChatOllama(
    model="qc:latest",
    temperature=0,
).bind_tools(tools)

# A second model binding for memory summarization (can reuse same model)
memory_model = ChatOllama(model="qc:latest", temperature=0)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a tool-using AI assistant.\n"
        "Use PythonREPLTool for any computation.\n"
        "Use DuckDuckGoSearchRun for any web/current info.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct.\n"
        "When you are done, give the final answer directly."
    )
)

# Optional: memory prompt
MEMORY_PROMPT = SystemMessage(
    content=(
        "Summarize the key facts, decisions, and intermediate results so far.\n"
        "Write it as short 'working memory' that helps continue the task.\n"
        "Do NOT include tool output verbatim; compress it.\n"
        "If there is no useful info yet, return an empty string."
    )
)

# Optional: reflection prompt
REFLECT_PROMPT = SystemMessage(
    content=(
        "Check whether the assistant has fully answered the user's request.\n"
        "If something is missing, say what tool call should happen next.\n"
        "Otherwise respond with: OK"
    )
)


# ----------------------------
# 3) Helpers
# ----------------------------
def _normalize_tool_args(args: Any):
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        if "__arg1" in args:
            return args["__arg1"]
        if len(args) == 1:
            return next(iter(args.values()))
        return args
    return args


def _tool_signature(tool_call: Dict[str, Any]) -> str:
    """Create a stable signature to detect repetition/stuck loops."""
    name = tool_call.get("name", "")
    args = tool_call.get("args", "")
    # normalize to something hashable-ish
    if isinstance(args, dict):
        args = json.dumps(args, sort_keys=True)
    return f"{name}:{args}"


def _rolling_append(lst: Optional[List[str]], item: str, limit: int) -> List[str]:
    lst = list(lst or [])
    lst.append(item)
    if len(lst) > limit:
        lst = lst[-limit:]
    return lst


# ----------------------------
# 4) Nodes
# ----------------------------
def call_model(state: AgentState, config: RunnableConfig):
    """LLM step: uses system prompt + memory + messages."""
    step = int(state.get("step", 0)) + 1

    memory = (state.get("memory") or "").strip()
    memory_msg = SystemMessage(content=f"WORKING_MEMORY:\n{memory}") if memory else None

    prompt_msgs = [SYSTEM_PROMPT]
    if memory_msg:
        prompt_msgs.append(memory_msg)
    prompt_msgs.extend(list(state["messages"]))

    response = model.invoke(prompt_msgs, config=config)

    return {
        "messages": [response],
        "step": step,
    }


def tool_node(state: AgentState):
    """Execute tool calls from last AI message. Includes retries + repetition tracking."""
    last = state["messages"][-1]
    outputs: List[ToolMessage] = []

    tool_calls = getattr(last, "tool_calls", None) or []
    last_sigs = state.get("last_tool_signatures") or []

    for tc in tool_calls:
        sig = _tool_signature(tc)
        last_sigs = _rolling_append(last_sigs, sig, REPEAT_STUCK_WINDOW)

        name = tc["name"]
        tool = tools_by_name.get(name)
        if tool is None:
            outputs.append(
                ToolMessage(
                    content=f"Unknown tool: {name}",
                    name=name,
                    tool_call_id=tc.get("id", ""),
                )
            )
            continue

        tool_args = _normalize_tool_args(tc.get("args"))

        # Simple retry loop for reliability
        result: str = ""
        last_err: Optional[Exception] = None
        for attempt in range(MAX_TOOL_RETRIES + 1):
            try:
                r = tool.invoke(tool_args)
                if not isinstance(r, str):
                    r = json.dumps(r, ensure_ascii=False)
                result = r
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None and not result:
            result = f"Tool error (after retries): {type(last_err).__name__}: {last_err}"

        outputs.append(
            ToolMessage(
                content=result,
                name=name,
                tool_call_id=tc.get("id", ""),
            )
        )

    return {
        "messages": outputs,
        "last_tool_signatures": last_sigs,
    }


def summarize_memory_node(state: AgentState, config: RunnableConfig):
    """Occasionally compress history into memory."""
    # Keep memory summarization deterministic by using temp=0 model and a stable prompt.
    msgs = list(state["messages"])
    response = memory_model.invoke([MEMORY_PROMPT] + msgs, config=config)

    memory_text = (response.content or "").strip()
    return {"memory": memory_text}


def reflect_node(state: AgentState, config: RunnableConfig):
    """
    Optional verification step:
    - If reflection says OK -> end
    - Else -> continue (agent will likely do more tool calls)
    """
    msgs = list(state["messages"])
    response = memory_model.invoke([REFLECT_PROMPT] + msgs, config=config)
    return {"messages": [AIMessage(content=f"[REFLECT] {response.content.strip()}")]}


# ----------------------------
# 5) Routing / stopping rules
# ----------------------------
def should_continue(state: AgentState):
    # Hard stop: max steps
    step = int(state.get("step", 0))
    max_steps = int(state.get("max_steps", MAX_STEPS_DEFAULT))
    if step >= max_steps:
        return "end_max_steps"

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    # If the model requested tools, check for "stuck" repetition patterns
    if tool_calls:
        sigs = state.get("last_tool_signatures") or []
        if len(sigs) == REPEAT_STUCK_WINDOW and len(set(sigs)) == 1:
            return "end_stuck"
        return "tools"

    # No tool calls: optionally reflect/verify before ending
    return "reflect"


def should_summarize_memory(state: AgentState):
    step = int(state.get("step", 0))
    # summarize every N steps (and only after we have enough context)
    if step > 0 and step % MEMORY_EVERY == 0:
        return "summarize"
    return "skip"


def after_reflect(state: AgentState):
    """
    If reflection says OK -> end.
    Else -> keep going.
    """
    last = state["messages"][-1]
    text = (last.content or "")
    if "OK" in text:
        return "end"
    return "agent"


# ----------------------------
# 6) Build graph (multi-stage flow)
# ----------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("agent", call_model)
    g.add_node("tools", tool_node)
    g.add_node("memory", summarize_memory_node)
    g.add_node("reflect", reflect_node)

    g.set_entry_point("agent")

    # After agent -> either tools or reflect (or end conditions)
    g.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "reflect": "reflect",
            "end_max_steps": END,
            "end_stuck": END,
        },
    )

    # After tools -> maybe summarize memory -> back to agent
    g.add_conditional_edges(
        "tools",
        should_summarize_memory,
        {"summarize": "memory", "skip": "agent"},
    )
    g.add_edge("memory", "agent")

    # After reflect -> decide end or continue
    g.add_conditional_edges(
        "reflect",
        after_reflect,
        {"end": END, "agent": "agent"},
    )

    return g.compile()


# ----------------------------
# 7) Run with deterministic streaming
# ----------------------------
def main():
    graph = build_graph()

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk net worth' "
        "and summarize in 1 line."
    )

    inputs: AgentState = {
        "messages": [HumanMessage(content=query)],
        "step": 0,
        "max_steps": 12,
        "memory": "",
        "last_tool_signatures": [],
        "stop_reason": "",
    }

    # Deterministic inspection: print every step's last message type + summary
    for state in graph.stream(inputs, stream_mode="values"):
        last = state["messages"][-1]
        cls = last.__class__.__name__
        content = (last.content or "")
        step_num = state.get("step", 0)
        preview = content[:200].replace("\n", " ")
        print(f"[step={step_num:02d}] {cls}: {preview}")



    # Final: last message
    final_state = graph.invoke(inputs)
    print("\nFINAL:", final_state["messages"][-1].content)


if __name__ == "__main__":
    main()
