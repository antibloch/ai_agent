import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ----------------------------
# Tools
# ----------------------------
def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


# ----------------------------
# Reflexion Orchestrator
# ----------------------------
class ReflexionAgent:
    def __init__(self, model, tools, max_attempts: int = 2):
        self.model = model
        self.tools = tools
        self.max_attempts = max_attempts

        self.base_system_prompt = (
            "You are a tool-using AI assistant.\n"
            "Use PythonREPLTool for computation.\n"
            "Use DuckDuckGoSearchRun for web/current info.\n"
            "Be concise and correct.\n"
            "If you did not call a tool, DO NOT claim you did."
        )

        self.critique_prompt = (
            "You are a strict reviewer.\n"
            "Critically analyze the previous answer.\n"
            "Check for:\n"
            "- Missing tool usage\n"
            "- Incorrect computation\n"
            "- Outdated info\n"
            "- Hallucinations\n"
            "If everything is correct, reply ONLY with: CORRECT.\n"
            "Otherwise explain briefly what is wrong."
        )

        self.reflection_memory: List[str] = []

    def _run_attempt(self, query: str, reflection: str = "") -> Dict[str, Any]:
        system_prompt = self.base_system_prompt

        if reflection:
            system_prompt += f"\n\nPrevious failure notes:\n{reflection}\n"
            system_prompt += "Fix the issues above."

        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
        )

        return agent.invoke({"messages": [HumanMessage(content=query)]})

    def _critique(self, answer: str) -> str:
        critique_messages = [
            SystemMessage(content=self.critique_prompt),
            HumanMessage(content=answer),
        ]

        critique = self.model.invoke(critique_messages)
        return critique.content.strip()

    def invoke(self, query: str) -> Dict[str, Any]:
        reflection = ""
        last_result = None

        for attempt in range(self.max_attempts):
            print(f"\n--- Attempt {attempt+1} ---")

            result = self._run_attempt(query, reflection)
            answer = result["messages"][-1].content

            critique = self._critique(answer)

            if critique == "CORRECT.":
                print("Answer validated.")
                return result

            print("Critique:", critique)
            reflection = critique
            last_result = result

        print("Max attempts reached. Returning last result.")
        return last_result


# ----------------------------
# Main
# ----------------------------
def main():
    tools = build_tools()

    model = ChatOllama(
        model="qc:latest",
        temperature=0,
    )

    reflex_agent = ReflexionAgent(model=model, tools=tools, max_attempts=2)

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk's net worth' "
        "and summarize in 1 line."
    )

    result = reflex_agent.invoke(query)
    print("\nFinal Answer:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
