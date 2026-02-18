import os
from typing import Any, Dict, List, TypedDict, Tuple

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

from langchain.agents import create_agent  # LangChain v1 loop agent (we'll use it as the *executor*)
from langgraph.graph import StateGraph, END


os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


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
        description="A short ordered list of concrete steps to solve the user's request."
    )


# ----------------------------
# 3) LangGraph state
# ----------------------------
class AgentState(TypedDict):
    user_query: str
    plan: List[str]
    completed: List[Tuple[str, str]]  # (step, observation/result)
    final_answer: str


# ----------------------------
# 4) Nodes
# ----------------------------
def planner_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    """
    Create an explicit plan. No tools here — just decomposition.
    """
    parser = PydanticOutputParser(pydantic_object=Plan)

    system = SystemMessage(
        content=(
            "You are a planner. Break the user's request into a short sequence of steps.\n"
            "Rules:\n"
            "- Output ONLY valid JSON that matches the schema.\n"
            "- Steps should be actionable and reference tools if relevant (search, python), but do NOT run tools.\n"
            f"{parser.get_format_instructions()}"
        )
    )

    msg = HumanMessage(content=state["user_query"])
    raw = model.invoke([system, msg]).content

    plan_obj = parser.parse(raw)
    return {"plan": plan_obj.steps, "completed": []}


def executor_node(state: AgentState, model: ChatOllama, tools) -> Dict[str, Any]:
    """
    Execute exactly ONE step from the plan using a tool-using agent.
    """
    if not state["plan"]:
        return {}

    step = state["plan"][0]
    remaining = state["plan"][1:]

    # Executor agent: still ReAct internally, but constrained to *this step only*.
    exec_system_prompt = (
        "You are an execution agent.\n"
        "You will be given ONE step from a plan.\n"
        "Do whatever tool use is necessary to complete that step.\n"
        "Return a concise result for that step only.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "\n"
        "Tooling rules:\n"
        "- Use PythonREPLTool for computations.\n"
        "- Use DuckDuckGoSearchRun for current web info.\n"
    )

    exec_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=exec_system_prompt,
    )

    result: Dict[str, Any] = exec_agent.invoke(
        {"messages": [HumanMessage(content=f"PLAN STEP:\n{step}")]}
    )
    step_output = result["messages"][-1].content

    completed = state["completed"] + [(step, step_output)]
    return {"plan": remaining, "completed": completed}


def should_continue(state: AgentState) -> str:
    return "execute" if state["plan"] else "finalize"


def finalizer_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    """
    Synthesize the final response from the executed steps.
    """
    # Build a compact summary for the finalizer
    executed = "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in state["completed"]])

    system = SystemMessage(
        content=(
            "You are a finalizer. Produce the final answer to the user.\n"
            "Use ONLY the step results provided. If something is missing, say so.\n"
            "Keep it concise.\n"
        )
    )

    msg = HumanMessage(
        content=(
            f"USER QUERY:\n{state['user_query']}\n\n"
            f"EXECUTED STEPS + RESULTS:\n{executed}\n\n"
            "FINAL ANSWER:"
        )
    )

    final = model.invoke([system, msg]).content
    return {"final_answer": final}


# ----------------------------
# 5) Build graph
# ----------------------------
def build_plan_execute_graph(model: ChatOllama, tools):
    g = StateGraph(AgentState)

    g.add_node("plan", lambda s: planner_node(s, model))
    g.add_node("execute", lambda s: executor_node(s, model, tools))
    g.add_node("finalize", lambda s: finalizer_node(s, model))

    g.set_entry_point("plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", should_continue, {"execute": "execute", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g.compile()


# ----------------------------
# 6) Main
# ----------------------------
def main():
    tools = build_tools()
    
    model = ChatOllama(
        model="qc:latest",
        temperature=0,
    )

    app = build_plan_execute_graph(model, tools)

    # query = (
    #     "Compute pi upto 13 decimal places using python, "
    #     "then search the web for 'latest Elon Musk's net worth' "
    #     "and summarize in 1 line."
    # )

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'Elon Musk's residence' "
        "and then find the latest news at Elon Musk's residence. "
        "and summarize all this in 1 line."
    )

    state: AgentState = {
        "user_query": query,
        "plan": [],
        "completed": [],
        "final_answer": "",
    }

    out = app.invoke(state)
    print(out["final_answer"])


if __name__ == "__main__":
    main()
