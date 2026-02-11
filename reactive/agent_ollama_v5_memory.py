import os
from typing import Any, Dict

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

from langgraph.checkpoint.memory import InMemorySaver

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


def main():
    tools = build_tools()

    # Same base model instance used for both normal turns AND summarization
    base_model = ChatOllama(
        model="qc:latest",
        temperature=0,
    )

    system_prompt = (
        "You are a tool-using AI assistant.\n"
        "Use PythonREPLTool for any computation.\n"
        "Use DuckDuckGoSearchRun for any web/current info.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct."
    )

    checkpointer = InMemorySaver()

    # Summarize when history gets large; keep last N messages verbatim
    # IMPORTANT: You asked for the “same base model”, so we pass base_model here too.
    summarizer = SummarizationMiddleware(
        model=base_model,
        trigger=("tokens", 3000),   # adjust to your model/context
        keep=("messages", 20),      # keep last 20 messages un-summarized
    )

    agent = create_agent(
        model=base_model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[summarizer],
        checkpointer=checkpointer,
    )

    # Thread memory lives under this ID:
    config: RunnableConfig = {"configurable": {"thread_id": "demo-thread-1"}}

    # Multiple calls accumulate memory in this thread:
    agent.invoke({"messages": [HumanMessage(content="Hi, my name is Bob. Remember it.")]}, config)

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk's net worth' "
        "and summarize in 1 line."
    )
    result: Dict[str, Any] = agent.invoke({"messages": [HumanMessage(content=query)]}, config)
    print(result["messages"][-1].content)

    # Proof memory works (even if summarized later):
    result2 = agent.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
    print(result2["messages"][-1].content)


if __name__ == "__main__":
    main()
