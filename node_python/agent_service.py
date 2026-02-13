import os
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from langchain_ollama import ChatOllama  # <- Ollama chat model

import requests
import json
from langchain_core.tools import Tool

def build_node_stats_tool():
    BASE_URL = "http://localhost:3000"

    def call_node_stats(query: str) -> str:
        """
        Calls Node.js GET /api/stats?q=...
        """
        try:
            r = requests.get(
                f"{BASE_URL}/api/stats",
                params={"q": query},
                timeout=5,
            )
            r.raise_for_status()
            return json.dumps(r.json())
        except requests.RequestException as e:
            return json.dumps({"error": str(e)})

    return Tool(
        name="get_charity_stats",
        description=(
            "Fetch internal charity statistics from the Node.js service. "
            "Input should be any string. Returns JSON with statistics."
        ),
        func=call_node_stats,
    )



def build_tools():
    return [
        build_node_stats_tool(),  # <-- new tool
        DuckDuckGoSearchRun(),
        PythonREPLTool(),
    ]


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
        "Use `get_charity_stats` to fetch internal statistics.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct."
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,  # NOTE: v1 uses system_prompt (not prompt)
    )

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk's net worth' "
        " then search internal charity stats with query 'total charities' "
        "and summarize in 1 line."
    )

    result: Dict[str, Any] = agent.invoke({"messages": [HumanMessage(content=query)]})

    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
