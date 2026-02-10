import os
from typing import Any, Dict, List, Tuple

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from langchain_openai import ChatOpenAI  # tool-calling capable

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"
# export OPENAI_API_KEY=...

def build_tools():
    # DuckDuckGoSearchRun: expects a search query string (single input tool)
    search = DuckDuckGoSearchRun()
    # PythonREPLTool: expects python code string (single input tool)
    py = PythonREPLTool()
    return [search, py]

def main():
    tools = build_tools()

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    system_prompt = (
        "You are a tool-using assistant.\n"
        "Use PythonREPLTool for any computations.\n"
        "Use DuckDuckGoSearchRun for any web/current-events lookup.\n"
        "Be concise.\n"
    )

    agent = create_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,   # system prompt string
        debug=True,
        name="python_search_agent",
    )

    query = (
        "Compute (5+3)*2 using python, then search the web for "
        "'LangChain AgentExecutor import error' and summarize in 1 line."
    )

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # result is an AgentState-like dict; final messages contain the answer
    final_msg = result["messages"][-1]
    print(final_msg.content)

if __name__ == "__main__":
    main()
