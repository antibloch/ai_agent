import os
from typing import Any, Dict, List

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


def _count_chars(messages: List[Any]) -> int:
    # crude proxy if you don’t have token counter utilities available
    total = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            total += len(content)
    return total


def main():
    tools = build_tools()

    base_model = ChatOllama(model="qc:latest", temperature=0)

    system_prompt = (
        "You are a tool-using AI assistant.\n"
        "Use PythonREPLTool for any computation.\n"
        "Use DuckDuckGoSearchRun for any web/current info.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct."
    )

    checkpointer = InMemorySaver()

    # Tune these:
    MAX_HISTORY_CHARS = 12000   # when exceeded -> summarize
    KEEP_LAST_N = 12            # keep last N messages verbatim

    @before_model
    def summarize_if_needed(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        if len(messages) <= KEEP_LAST_N:
            return None

        if _count_chars(messages) < MAX_HISTORY_CHARS:
            return None

        # Split: keep the last N messages, summarize everything before that.
        head = messages[:-KEEP_LAST_N]
        tail = messages[-KEEP_LAST_N:]

        # Build a summarization prompt using the SAME base model.
        # Keep summary factual and compact, focusing on user facts, goals, decisions, tool results.
        summarization_prompt = [
            SystemMessage(content=(
                "Summarize the conversation so far into a compact memory.\n"
                "Rules:\n"
                "- Keep only durable facts (names, preferences, goals, constraints, decisions, important tool results).\n"
                "- Drop chit-chat and transient details.\n"
                "- Write in 6-12 bullet points.\n"
            )),
            HumanMessage(content="\n\n".join(
                f"{getattr(m, 'type', m.__class__.__name__)}: {getattr(m, 'content', '')}"
                for m in head
            )),
        ]

        summary_msg = base_model.invoke(summarization_prompt)
        summary_text = getattr(summary_msg, "content", "").strip()
        if not summary_text:
            return None

        # Replace entire message list with:
        # - original system prompt (as SystemMessage)
        # - a synthetic "memory summary" message
        # - the most recent tail
        new_messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=f"[Memory summary]\n{summary_text}"),
            *tail,
        ]

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }

    agent = create_agent(
        model=base_model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[summarize_if_needed],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "demo-thread-1"}}

    agent.invoke({"messages": [HumanMessage(content="Hi, my name is Bob. Remember it.")]}, config)

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'latest Elon Musk's net worth' "
        "and summarize in 1 line."
    )
    result: Dict[str, Any] = agent.invoke({"messages": [HumanMessage(content=query)]}, config)
    print(result["messages"][-1].content)

    result2 = agent.invoke({"messages": [HumanMessage(content="summarize previous conversation and let me know latest news in AI by searching the web")]}, config)
    print(result2["messages"][-1].content)


if __name__ == "__main__":
    main()
