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
    # fixed
    user_query: str

    # outer reflexion loop
    attempt: int
    max_attempts: int
    critique: str          # latest critique
    reflection: str        # critique notes injected into planning
    last_final_output: str # final output from previous attempt (attempt > 1)

    # inner plan-execute
    plan: List[str]
    completed: List[Tuple[str, str]]  # (step, output)

    # executor-produced final output (output of the last plan step)
    final_output: str


# ----------------------------
# 4) Nodes
# ----------------------------
def planner_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    """
    Produce a plan as JSON.

    Attempt 1:
      - Planner sees user query + planning rules.

    Attempt > 1:
      - Planner sees a structured case file:
          1) original user query
          2) what the last attempt did (step outputs + final output)
          3) critique explaining what went wrong
      - Then it produces a revised plan.
    """
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

    msg = HumanMessage(content=human_content)

    raw = model.invoke([system, msg]).content
    plan_obj = parser.parse(raw)

    return {"plan": plan_obj.steps, "completed": [], "final_output": ""}


def executor_one_step_node(state: AgentState, model: ChatOllama, tools) -> Dict[str, Any]:
    """
    Executes exactly ONE step.
    If this was the last step, we store it as final_output.
    Crucially, the executor always sees the USER QUERY to stay grounded.
    """
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

    exec_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=exec_system_prompt,
    )

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
        update["final_output"] = step_output

    return update


def should_continue_execute(state: AgentState) -> str:
    return "execute" if (state.get("plan") or []) else "critique"


def critique_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    """
    Critique compares USER QUERY vs executor-produced final_output.
    Includes plan and step outputs for grounding.
    """
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


def should_retry(state: AgentState) -> str:
    """
    Retry conditions:
      - If final output is empty / "no context" / "not provided" => retry (if attempts remain)
      - Else, follow critique verdict
    """
    attempt = int(state.get("attempt", 1))
    max_attempts = int(state.get("max_attempts", 2))

    final_output = (state.get("final_output") or "").strip()
    if is_bad_final_output(final_output):
        return "retry" if attempt < max_attempts else "end"

    critique = (state.get("critique") or "").strip()
    if is_critique_correct(critique):
        return "end"

    if attempt >= max_attempts:
        return "end"

    return "retry"


def reset_for_retry_node(state: AgentState) -> Dict[str, Any]:
    """
    Prepare for next attempt:
      - attempt++
      - reflection = critique
      - last_final_output = final_output
      - keep completed from last attempt so planner can see what happened
        NOTE: planner_node reads completed and then resets completed to [].
    """
    next_attempt = int(state.get("attempt", 1)) + 1

    return {
        "attempt": next_attempt,
        "reflection": (state.get("critique") or "").strip(),
        "last_final_output": (state.get("final_output") or "").strip(),
        "plan": [],
        "final_output": "",
        # DO NOT clear "completed" here; planner will use it as case file input
    }


# ----------------------------
# 5) Build graph
# ----------------------------
def build_plan_execute_reflexion_graph(model: ChatOllama, tools):
    g = StateGraph(AgentState)

    g.add_node("plan", lambda s: planner_node(s, model))
    g.add_node("execute", lambda s: executor_one_step_node(s, model, tools))
    g.add_node("critique", lambda s: critique_node(s, model))
    g.add_node("reset_for_retry", reset_for_retry_node)

    g.set_entry_point("plan")

    # plan -> execute loop
    g.add_edge("plan", "execute")
    g.add_conditional_edges(
        "execute",
        should_continue_execute,
        {"execute": "execute", "critique": "critique"},
    )

    # critique -> retry or end
    g.add_conditional_edges(
        "critique",
        should_retry,
        {"retry": "reset_for_retry", "end": END},
    )
    g.add_edge("reset_for_retry", "plan")

    return g.compile()


# ----------------------------
# 6) Run
# ----------------------------
def main():
    tools = build_tools()

    model = ChatOllama(
        model="qc:latest",
        temperature=0,
    )

    app = build_plan_execute_reflexion_graph(model, tools)

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'Elon Musk's residence' "
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
    }

    out = app.invoke(state)

    print("\nFINAL OUTPUT (executor-produced):\n")
    print((out.get("final_output") or "").strip())

    print("\nCRITIQUE:\n")
    print((out.get("critique") or "").strip())

    print("\nATTEMPTS USED:\n")
    print(out.get("attempt"))


if __name__ == "__main__":
    main()
