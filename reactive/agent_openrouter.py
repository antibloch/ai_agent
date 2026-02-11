import os
from typing import Any, Dict, List, Tuple

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from langchain_openai import ChatOpenAI  # tool-calling capable

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"
os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]  # or set directly
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# export OPENAI_API_KEY=...

def build_tools():
    # DuckDuckGoSearchRun: expects a search query string (single input tool)
    search = DuckDuckGoSearchRun()
    # PythonREPLTool: expects python code string (single input tool)
    py = PythonREPLTool()
    return [search, py]

def main():
    tools = build_tools()

    # model = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0.0,
    # )

    model = ChatOpenAI(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
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
        system_prompt=system_prompt,
        debug=True,
        name="python_search_agent",
        )


    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk's net worth' "
        "and summarize in 1 line."
    )

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # result is an AgentState-like dict; final messages contain the answer
    final_msg = result["messages"][-1]
    print("..................................")
    print(final_msg.content)

if __name__ == "__main__":
    main()
