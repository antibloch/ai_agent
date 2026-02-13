import os
import re
from typing import Any, Dict, List, Tuple, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

from langchain.agents import create_agent  # executor loop agent
from langgraph.graph import StateGraph, END


os.environ["USER_AGENT"] = "my-langchain-agent/1.0"

# ----------------------------
# 0) Critique helpers
# ----------------------------
_CORRECT_RE = re.compile(r"^\s*CORRECT\s*[\.\!\u2705✅]*\s*$", re.IGNORECASE)
_BAD_FINAL_RE = re.compile(
    r"(no\s+context|not\s+provided|no\s+prior\s+context|cannot\s+answer|insufficient\s+information)",
    re.IGNORECASE,
)


def is_critique_correct(text: str) -> bool:
    return bool(_CORRECT_RE.match((text or "").strip()))


def is_bad_final_output(text: str) -> bool:
    return bool(_BAD_FINAL_RE.search((text or "").strip())) or not (text or "").strip()


# Strong critique prompt (same as before)
CRITIQUE_PROMPT = (
    "You are a strict reviewer.\n"
    "Decide if FINAL OUTPUT correctly answers USER QUERY.\n"
    "Reply ONLY with CORRECT or a short critique.\n\n"
    "Automatic NOT CORRECT if any of these hold:\n"
    "- FINAL OUTPUT says or implies missing context/input (e.g., 'no context', 'not provided')\n"
    "- Any required part of the USER QUERY is missing\n"
    "- USER asked for 1 line but FINAL OUTPUT has multiple lines\n"
    "- USER asked for current/news info but there is no evidence of web info in STEP OUTPUTS\n"
    "- FINAL OUTPUT includes disclaimers instead of completing the task\n\n"
    "Now evaluate using:\n"
    "- USER QUERY\n"
    "- PLAN EXECUTED\n"
    "- STEP OUTPUTS\n"
    "- FINAL OUTPUT\n\n"
    "If correct, reply ONLY: CORRECT.\n"
    "Otherwise, explain briefly what is missing/wrong and what to fix.\n"
)

# NEW: finalizer system prompt
FINALIZER_SYSTEM_PROMPT = (
    "You are a formatting finalizer.\n"
    "You will be given:\n"
    "1) the original USER QUERY (which includes formatting requirements)\n"
    "2) a RAW RESPONSE produced by another assistant\n\n"
    "Your job: rewrite the RAW RESPONSE so it strictly satisfies the USER QUERY requirements.\n"
    "Rules:\n"
    "- Do NOT add new facts.\n"
    "- Do NOT claim tools were used.\n"
    "- Preserve meaning; only fix formatting, ordering, brevity, and remove extraneous labels/preambles.\n"
    "- If the USER QUERY requires '1 line', output EXACTLY one line.\n"
    "- If required items are missing in RAW RESPONSE, keep it transparent (e.g., 'Missing: ...').\n"
    "Output ONLY the final formatted answer (no 'FINAL:', no explanations)."
)

# ----------------------------
# 1) Tools
# ----------------------------
def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


# ----------------------------
# 2) Planner schema
# ----------------------------
class Plan(BaseModel):
    steps: List[str] = Field(
        description=(
            "A short ordered list of concrete steps to solve the user's request. "
            "The last step MUST produce the final user-facing answer in the requested format."
        )
    )


# ----------------------------
# 3) LangGraph state
# ----------------------------
class AgentState(TypedDict):
    user_query: str

    attempt: int
    max_attempts: int
    critique: str
    reflection: str
    last_final_output: str

    plan: List[str]
    completed: List[Tuple[str, str]]

    # from executor last step (raw)
    final_output: str

    # NEW: final formatted output (only produced after CORRECT)
    formatted_output: str


# ----------------------------
# 4) Nodes
# ----------------------------
def planner_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=Plan)

    attempt = int(state.get("attempt", 1))
    user_query = state["user_query"]

    reflection = (state.get("reflection") or "").strip()
    last_final = (state.get("last_final_output") or "").strip()
    prev_steps = state.get("completed") or []

    case_file = ""
    if attempt > 1:
        if prev_steps:
            steps_block = "\n".join(
                [f"{i+1}. {s}\n   Output: {o}" for i, (s, o) in enumerate(prev_steps)]
            )
        else:
            steps_block = "(no recorded step outputs)"

        case_file = (
            "CASE FILE (previous attempt)\n"
            "===========================\n"
            "ORIGINAL USER QUERY:\n"
            f"{user_query}\n\n"
            "LAST ATTEMPT STEP OUTPUTS:\n"
            f"{steps_block}\n\n"
            "LAST ATTEMPT FINAL OUTPUT:\n"
            f"{(last_final or '(none)')}\n\n"
            "CRITIQUE (what went wrong / what to fix):\n"
            f"{(reflection or '(none)')}\n"
            "===========================\n\n"
        )

    system_rules = (
        "You are a planner. Produce an improved step-by-step plan.\n\n"
        "Output rules:\n"
        "- Output ONLY valid JSON matching the schema.\n"
        "- Do NOT execute tools.\n\n"
        "Planning rules:\n"
        "- Cover ALL parts of the user request.\n"
        "- Keep steps minimal (usually 2-6).\n"
        "- Steps must be actionable and may reference tools (python, search) if relevant.\n"
        "- The LAST step MUST explicitly produce the final user-facing answer in the requested format.\n"
        "- The LAST step MUST list EXACTLY what to include in the final answer (required fields).\n"
        "- If the user asks for current/latest/news, include a search step and ensure the final step references its result.\n"
        "- If a previous attempt failed, adjust the plan to fix the critique.\n\n"
        f"{parser.get_format_instructions()}"
    )

    system = SystemMessage(content=system_rules)

    if attempt > 1:
        human_content = (
            f"{case_file}"
            "Now produce a revised plan that fixes the critique.\n"
            "Return JSON only.\n"
        )
    else:
        human_content = (
            "USER QUERY:\n"
            f"{user_query}\n\n"
            "Produce a plan.\n"
            "Return JSON only.\n"
        )

    raw = model.invoke([system, HumanMessage(content=human_content)]).content
    plan_obj = parser.parse(raw)

    return {
        "plan": plan_obj.steps,
        "completed": [],
        "final_output": "",
        "formatted_output": "",
    }


def executor_one_step_node(state: AgentState, model: ChatOllama, tools) -> Dict[str, Any]:
    plan = state.get("plan") or []
    if not plan:
        return {}

    step = plan[0]
    remaining = plan[1:]

    exec_system_prompt = (
        "You are an execution agent.\n"
        "You will be given the USER QUERY and ONE plan step.\n"
        "Always keep the USER QUERY in mind; the step must contribute toward it.\n"
        "Complete the step using tools if needed.\n"
        "Return only the result for this step.\n"
        "If you did not call a tool, DO NOT claim you did.\n\n"
        "Tooling:\n"
        "- Use PythonREPLTool for computation.\n"
        "- Use DuckDuckGoSearchRun for web/current info.\n"
        "- For web info steps: include 'SOURCE:' with the domain/outlet name in your step output.\n"
        "- If the step asks for the final answer: output EXACTLY one line, no extra labels.\n"
    )

    exec_agent = create_agent(model=model, tools=tools, system_prompt=exec_system_prompt)

    result: Dict[str, Any] = exec_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"USER QUERY:\n{state['user_query']}\n\n"
                        f"PLAN STEP:\n{step}\n\n"
                        "Return ONLY the result for this step."
                    )
                )
            ]
        }
    )

    step_output = result["messages"][-1].content
    completed = (state.get("completed") or []) + [(step, step_output)]
    update: Dict[str, Any] = {"plan": remaining, "completed": completed}

    if not remaining:
        update["final_output"] = step_output  # raw final output from executor

    return update


def should_continue_execute(state: AgentState) -> str:
    return "execute" if (state.get("plan") or []) else "critique"


def critique_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    user_query = state["user_query"]
    final_output = (state.get("final_output") or "").strip()
    completed = state.get("completed") or []

    plan_executed = "\n".join([f"{i+1}. {s}" for i, (s, _) in enumerate(completed)]) or "(none)"
    step_outputs = "\n".join([f"- Step: {s}\n  Output: {o}" for s, o in completed]) or "(none)"

    critique_input = (
        f"USER QUERY:\n{user_query}\n\n"
        f"PLAN EXECUTED:\n{plan_executed}\n\n"
        f"STEP OUTPUTS:\n{step_outputs}\n\n"
        f"FINAL OUTPUT (to user):\n{final_output}\n"
    )

    critique = model.invoke(
        [SystemMessage(content=CRITIQUE_PROMPT), HumanMessage(content=critique_input)]
    ).content.strip()

    return {"critique": critique}


def should_finalize_or_retry(state: AgentState) -> str:
    """
    Routing after critique:
      - If final output is obviously bad => retry (if attempts remain)
      - Else if critique CORRECT => go to finalizer
      - Else retry (if attempts remain)
      - Else end (best-effort)
    """
    attempt = int(state.get("attempt", 1))
    max_attempts = int(state.get("max_attempts", 2))

    final_output = (state.get("final_output") or "").strip()
    critique = (state.get("critique") or "").strip()

    # if is_bad_final_output(final_output):
    #     return "retry" if attempt < max_attempts else "end"

    if is_critique_correct(critique):
        return "finalize"

    return "retry" if attempt < max_attempts else "end"


def reset_for_retry_node(state: AgentState) -> Dict[str, Any]:
    next_attempt = int(state.get("attempt", 1)) + 1
    return {
        "attempt": next_attempt,
        "reflection": (state.get("critique") or "").strip(),
        "last_final_output": (state.get("final_output") or "").strip(),
        "plan": [],
        "final_output": "",
        "formatted_output": "",
        # DO NOT clear "completed" here; planner will read it as case file input
    }


def finalizer_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    """
    Runs ONLY after critique returns CORRECT.
    It formats the raw executor final_output according to user_query requirements,
    without adding new facts.
    """
    user_query = state["user_query"]
    raw_response = (state.get("final_output") or "").strip()

    messages = [
        SystemMessage(content=FINALIZER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"USER QUERY:\n{user_query}\n\n"
                f"RAW RESPONSE:\n{raw_response}\n"
            )
        ),
    ]

    formatted = model.invoke(messages).content.strip()

    # Safety: if finalizer accidentally outputs multiple lines but user asked 1 line, force single line.
    if "1 line" in user_query.lower() or "one line" in user_query.lower():
        formatted = " ".join([ln.strip() for ln in formatted.splitlines() if ln.strip()])

    return {"formatted_output": formatted}


# ----------------------------
# 5) Build graph
# ----------------------------
def build_plan_execute_reflexion_graph(model: ChatOllama, tools):
    g = StateGraph(AgentState)

    g.add_node("plan", lambda s: planner_node(s, model))
    g.add_node("execute", lambda s: executor_one_step_node(s, model, tools))
    g.add_node("critique", lambda s: critique_node(s, model))
    g.add_node("reset_for_retry", reset_for_retry_node)
    g.add_node("finalize", lambda s: finalizer_node(s, model))

    g.set_entry_point("plan")

    g.add_edge("plan", "execute")
    g.add_conditional_edges(
        "execute",
        should_continue_execute,
        {"execute": "execute", "critique": "critique"},
    )

    # critique -> finalize/retry/end
    g.add_conditional_edges(
        "critique",
        should_finalize_or_retry,
        {"finalize": "finalize", "retry": "reset_for_retry", "end": END},
    )

    g.add_edge("reset_for_retry", "plan")
    g.add_edge("finalize", END)

    return g.compile()


# ----------------------------
# 6) Run
# ----------------------------
def main():
    tools = build_tools()

    model = ChatOllama(model="qc:latest", temperature=0)

    app = build_plan_execute_reflexion_graph(model, tools)

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then find out about 'Elon Musk's residence' "
        "and then find the latest news at Elon Musk's residence, "
        "and summarize all this in 1 line."
    )

    state: AgentState = {
        "user_query": query,
        "attempt": 1,
        "max_attempts": 3,
        "critique": "",
        "reflection": "",
        "last_final_output": "",
        "plan": [],
        "completed": [],
        "final_output": "",
        "formatted_output": "",
    }

    out = app.invoke(state)

    print("\nRAW FINAL OUTPUT (executor-produced):\n")
    print((out.get("final_output") or "").strip())

    print("\nFORMATTED FINAL OUTPUT (finalizer):\n")
    # If critique never became CORRECT, formatted_output may be empty.
    # In that case, fall back to raw.
    formatted = (out.get("formatted_output") or "").strip()
    if formatted:
        print(formatted)
    else:
        print((out.get("final_output") or "").strip())

    print("\nCRITIQUE:\n")
    print((out.get("critique") or "").strip())

    print("\nATTEMPTS USED:\n")
    print(out.get("attempt"))


if __name__ == "__main__":
    main()
