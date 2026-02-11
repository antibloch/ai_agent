import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

from langchain.agents import create_agent
from langgraph.graph import StateGraph, END


os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ----------------------------
# 1) Tools
# ----------------------------
def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


# ----------------------------
# 2) ToT schemas
# ----------------------------
class CandidateSteps(BaseModel):
    candidates: List[str] = Field(
        description="A list of candidate next steps to take next. Each step should be a single actionable instruction."
    )


class ScoreObj(BaseModel):
    score: int = Field(description="Integer score 0..10 (10 is best).")
    reason: str = Field(description="1-2 lines why this candidate is good/bad.")


# ----------------------------
# 3) Search node structure
# ----------------------------
@dataclass
class Node:
    remaining_goal: str
    trace: List[Tuple[str, str]]  # (action_step, observation/result)
    score: float = 0.0
    depth: int = 0


# ----------------------------
# 4) LangGraph state
# ----------------------------
class AgentState(TypedDict):
    user_query: str
    best_trace: List[Tuple[str, str]]
    best_score: float
    final_answer: str


# ----------------------------
# 5) Helper: execute ONE step with tools (ReAct internally, but used as a primitive)
# ----------------------------
def execute_step(step: str, exec_model: ChatOllama, tools) -> str:
    exec_system_prompt = (
        "You are an execution agent.\n"
        "You will be given ONE step from a reasoning process.\n"
        "Use tools if needed. Return only the result of completing that step.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "\n"
        "Tooling rules:\n"
        "- Use PythonREPLTool for computations.\n"
        "- Use DuckDuckGoSearchRun for current web info.\n"
    )

    agent = create_agent(model=exec_model, tools=tools, system_prompt=exec_system_prompt)
    out: Dict[str, Any] = agent.invoke({"messages": [HumanMessage(content=f"STEP:\n{step}") ]})
    return out["messages"][-1].content


# ----------------------------
# 6) Helper: generate K candidate next steps (branching)
# ----------------------------
def propose_candidates(
    proposal_model: ChatOllama,
    remaining_goal: str,
    trace: List[Tuple[str, str]],
    k: int,
) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=CandidateSteps)

    history = "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in trace]) or "(none yet)"

    system = SystemMessage(
        content=(
            "You are generating candidate next actions in a Tree-of-Thought search.\n"
            "Given the remaining goal and what has been done so far, propose diverse next steps.\n"
            "Rules:\n"
            f"- Output ONLY valid JSON.\n"
            f"- Propose exactly {k} candidates.\n"
            "- Each candidate must be a single concrete action (often a tool-using step).\n"
            "- Avoid repeating the same candidate with tiny wording changes.\n"
            f"{parser.get_format_instructions()}"
        )
    )

    msg = HumanMessage(
        content=(
            f"REMAINING GOAL:\n{remaining_goal}\n\n"
            f"WHAT HAS BEEN DONE:\n{history}\n\n"
            "PROPOSE NEXT STEPS:"
        )
    )

    raw = proposal_model.invoke([system, msg]).content
    obj = parser.parse(raw)
    # force exactly k (best effort)
    return obj.candidates[:k]


# ----------------------------
# 7) Helper: score a candidate next step (value function)
# ----------------------------
def score_candidate(
    scorer_model: ChatOllama,
    overall_query: str,
    remaining_goal: str,
    trace: List[Tuple[str, str]],
    candidate_step: str,
) -> ScoreObj:
    parser = PydanticOutputParser(pydantic_object=ScoreObj)
    history = "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in trace]) or "(none yet)"

    system = SystemMessage(
        content=(
            "You are a strict evaluator for Tree-of-Thought candidate steps.\n"
            "Score how useful this next step is for solving the overall user query.\n"
            "Scoring rubric (0..10):\n"
            "- 9-10: highly likely to make clear progress, minimal ambiguity\n"
            "- 6-8: reasonable progress but some ambiguity\n"
            "- 3-5: weak progress / indirect\n"
            "- 0-2: irrelevant / redundant / risky\n"
            "Return JSON only.\n"
            f"{parser.get_format_instructions()}"
        )
    )

    msg = HumanMessage(
        content=(
            f"OVERALL USER QUERY:\n{overall_query}\n\n"
            f"REMAINING GOAL:\n{remaining_goal}\n\n"
            f"HISTORY:\n{history}\n\n"
            f"CANDIDATE NEXT STEP:\n{candidate_step}\n\n"
            "SCORE:"
        )
    )

    raw = scorer_model.invoke([system, msg]).content
    return parser.parse(raw)


# ----------------------------
# 8) Helper: update remaining goal (simple heuristic)
# ----------------------------
def update_remaining_goal(overall_query: str, trace: List[Tuple[str, str]], last_step: str, last_result: str) -> str:
    # Keep it simple: we keep the original query as "remaining goal".
    # In more advanced ToT, you'd have the model rewrite what's left to do.
    return overall_query


# ----------------------------
# 9) ToT search node: beam search over thoughts
# ----------------------------
def tot_search_node(
    state: AgentState,
    proposal_model: ChatOllama,
    scorer_model: ChatOllama,
    exec_model: ChatOllama,
    tools,
    max_depth: int = 3,
    branch_k: int = 4,
    beam_b: int = 2,
) -> Dict[str, Any]:
    """
    Simple ToT via beam search:
    - Start with root node (no steps done)
    - For depth in [0..max_depth-1]:
        - Expand each beam node by proposing K candidates
        - Score candidates
        - Execute top candidates, creating children
        - Keep best B children as new beam
    - Choose best node and return its trace
    """
    query = state["user_query"]

    beam: List[Node] = [Node(remaining_goal=query, trace=[], score=0.0, depth=0)]
    best_node: Optional[Node] = None

    for depth in range(max_depth):
        children: List[Node] = []

        for node in beam:
            # 1) propose
            candidates = propose_candidates(
                proposal_model=proposal_model,
                remaining_goal=node.remaining_goal,
                trace=node.trace,
                k=branch_k,
            )

            # 2) score all candidates, pick top few to execute
            scored: List[Tuple[int, str]] = []
            for cand in candidates:
                s = score_candidate(
                    scorer_model=scorer_model,
                    overall_query=query,
                    remaining_goal=node.remaining_goal,
                    trace=node.trace,
                    candidate_step=cand,
                )
                scored.append((s.score, cand))

            scored.sort(key=lambda x: x[0], reverse=True)
            to_execute = scored[: max(1, beam_b)]  # execute at least 1 per node

            # 3) execute selected candidates -> create child nodes
            for cand_score, cand_step in to_execute:
                obs = execute_step(cand_step, exec_model=exec_model, tools=tools)
                new_trace = node.trace + [(cand_step, obs)]
                new_remaining = update_remaining_goal(query, new_trace, cand_step, obs)
                child = Node(
                    remaining_goal=new_remaining,
                    trace=new_trace,
                    score=node.score + cand_score,
                    depth=node.depth + 1,
                )
                children.append(child)

        # 4) keep best beam_b children
        children.sort(key=lambda n: n.score, reverse=True)
        beam = children[:beam_b]

        if beam:
            if best_node is None or beam[0].score > best_node.score:
                best_node = beam[0]

        if not beam:
            break

    if best_node is None:
        return {"best_trace": [], "best_score": 0.0}

    return {"best_trace": best_node.trace, "best_score": best_node.score}


# ----------------------------
# 10) Finalizer: summarize best trace
# ----------------------------
def finalizer_node(state: AgentState, model: ChatOllama) -> Dict[str, Any]:
    executed = "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in state["best_trace"]]) or "(no steps)"
    system = SystemMessage(
        content=(
            "You are a finalizer. Produce a concise final answer to the user.\n"
            "Use ONLY the step results provided. If something is missing, say so.\n"
        )
    )
    msg = HumanMessage(
        content=(
            f"USER QUERY:\n{state['user_query']}\n\n"
            f"BEST TRACE (EXECUTED STEPS + RESULTS):\n{executed}\n\n"
            "FINAL ANSWER (1 line if user asked for 1 line):"
        )
    )
    final = model.invoke([system, msg]).content
    return {"final_answer": final}


# ----------------------------
# 11) Build graph
# ----------------------------
def build_tot_graph(proposal_model: ChatOllama, scorer_model: ChatOllama, exec_model: ChatOllama, tools):
    g = StateGraph(AgentState)

    g.add_node(
        "tot_search",
        lambda s: tot_search_node(
            s,
            proposal_model=proposal_model,
            scorer_model=scorer_model,
            exec_model=exec_model,
            tools=tools,
            max_depth=3,
            branch_k=4,
            beam_b=2,
        ),
    )
    g.add_node("finalize", lambda s: finalizer_node(s, scorer_model))

    g.set_entry_point("tot_search")
    g.add_edge("tot_search", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# ----------------------------
# 12) Main
# ----------------------------
def main():
    tools = build_tools()

    # Proposal model needs diversity -> higher temperature
    proposal_model = ChatOllama(model="qc:latest", temperature=0.7)
    # Scorer + executor deterministic
    scorer_model = ChatOllama(model="qc:latest", temperature=0.0)
    exec_model = ChatOllama(model="qc:latest", temperature=0.0)

    app = build_tot_graph(proposal_model, scorer_model, exec_model, tools)

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'Elon Musk's residence' "
        "and then find the latest news at Elon Musk's residence. "
        "and summarize all this in 1 line."
    )

    state: AgentState = {
        "user_query": query,
        "best_trace": [],
        "best_score": 0.0,
        "final_answer": "",
    }

    out = app.invoke(state)
    print(out["final_answer"])


if __name__ == "__main__":
    main()
