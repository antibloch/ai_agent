import os
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from langchain_ollama import ChatOllama  # <- Ollama chat model

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"

def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]

def main():
    tools = build_tools()

    # Use your local Ollama model
    model = ChatOllama(
        model="qc:latest",
        temperature=0,   # keep deterministic
        # base_url="http://localhost:11434",  # only if you changed Ollama host/port
    )

    system_prompt = (
        "You are a tool-using AI assistant.\n"
        "Use PythonREPLTool for any computation.\n"
        "Use DuckDuckGoSearchRun for any web/current info.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct."
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,  # NOTE: v1 uses system_prompt (not prompt)
    )

    # query = (
    #     "Compute pi upto 13 decimal places using python, "
    #     "then search the web for 'latest Elon Musk's net worth' "
    #     "and summarize in 1 line."
    # )

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then find out about 'Elon Musk's residence' "
        "and then find the latest news at Elon Musk's residence, "
        "and summarize all this in 1 line."
    )

    result: Dict[str, Any] = agent.invoke({"messages": [HumanMessage(content=query)]})

    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
