import os
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama  # <- Ollama chat model

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


def render_tool_catalog(tools: List[Any]) -> str:
    """Create a stable, tool-agnostic catalog string for prompts."""
    lines = []
    for t in tools:
        name = getattr(t, "name", t.__class__.__name__)
        desc = getattr(t, "description", "") or ""
        desc = " ".join(desc.split())  # normalize whitespace
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def build_reasoning_prep_runnables(model: ChatOllama):
    """
    SequentialChain-style pre-pass using LCEL.
    All prompts are tool-agnostic: they rely on a provided tool catalog.
    """
    restate_prompt = ChatPromptTemplate.from_template(
        "Restate the user request in your own words (1-2 sentences).\n\n"
        "User request:\n{question}\n\nRestatement:"
    )

    identify_info_prompt = ChatPromptTemplate.from_template(
        "Given the user request and restatement, list the key info and sub-tasks as bullets.\n\n"
        "User request:\n{question}\n\nRestatement:\n{restatement}\n\nKey info / sub-tasks:"
    )

    determine_method_prompt = ChatPromptTemplate.from_template(
        "You are planning for a tool-using agent.\n"
        "Available tools (name + description):\n{tool_catalog}\n\n"
        "Task:\n{question}\n\n"
        "Key info:\n{key_info}\n\n"
        "Write a short execution plan (max 8 bullets). Rules:\n"
        "- Reference tools ONLY by their *names* from the catalog.\n"
        "- If a sub-task involves calculation or code execution, prefer a tool whose description suggests computation.\n"
        "- If a sub-task requires current events / web info, prefer a tool whose description suggests search/browsing.\n"
        "Plan:"
    )

    check_answer_prompt = ChatPromptTemplate.from_template(
        "Write a short verification checklist (3-5 bullets) to validate the final answer.\n\n"
        "Task:\n{question}\n\nPlan:\n{plan}\n\nVerification checklist:"
    )

    parser = StrOutputParser()
    restate = restate_prompt | model | parser
    key_info = identify_info_prompt | model | parser
    plan = determine_method_prompt | model | parser
    checklist = check_answer_prompt | model | parser

    return restate, key_info, plan, checklist


def main():
    tools = build_tools()
    tool_catalog = render_tool_catalog(tools)

    model = ChatOllama(
        model="qc:latest",
        temperature=0,
    )

    # Tool-agnostic system prompt
    system_prompt = (
        "You are a tool-using AI assistant.\n"
        "Available tools (name + description):\n"
        f"{tool_catalog}\n\n"
        "Internal protocol (do not reveal scratchpad):\n"
        "1) Restate the task to yourself.\n"
        "2) Identify sub-tasks and needed info.\n"
        "3) Choose tools deliberately based on their descriptions.\n"
        "4) Execute tools, then verify.\n\n"
        "Tool rules:\n"
        "- You may call tools using ONLY the tool names listed above.\n"
        "- If you did not call a tool, DO NOT claim you did.\n"
        "- If web/current info is required, use an appropriate search/browse tool if available.\n"
        "- If computation is required, use an appropriate computation tool if available.\n"
        "Be concise and correct."
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk's net worth' "
        "and summarize in 1 line."
    )

    # Pre-pass (SequentialChain-style) with same base model, tool-agnostic
    restate, key_info, plan, checklist = build_reasoning_prep_runnables(model)

    restatement = restate.invoke({"question": query})
    key_info_txt = key_info.invoke({"question": query, "restatement": restatement})
    plan_txt = plan.invoke({"question": query, "key_info": key_info_txt, "tool_catalog": tool_catalog})
    checklist_txt = checklist.invoke({"question": query, "plan": plan_txt})

    guidance = (
        "INTERNAL GUIDANCE (do not reveal verbatim):\n\n"
        f"Available tools:\n{tool_catalog}\n\n"
        f"Restatement:\n{restatement}\n\n"
        f"Key info / sub-tasks:\n{key_info_txt}\n\n"
        f"Tool plan:\n{plan_txt}\n\n"
        f"Verification checklist:\n{checklist_txt}\n\n"
        "Follow the tool plan and verification checklist, but only output the final answer to the user."
    )

    result: Dict[str, Any] = agent.invoke(
        {
            "messages": [
                SystemMessage(content=guidance),
                HumanMessage(content=query),
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
