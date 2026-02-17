import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition  # <-- reference-style

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ----------------------------
# 1) Graph state (same)
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ----------------------------
# 2) Tools + model (same tools + same prompt + same bind_tools)
# ----------------------------
def build_tools():
    return [DuckDuckGoSearchRun(), PythonREPLTool()]


tools = build_tools()

model = ChatOllama(
    model="qc:latest",
    temperature=0,
    # base_url="http://localhost:11434",
).bind_tools(tools)  # IMPORTANT: enables tool calling


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a tool-using AI assistant.\n"
        "Use PythonREPLTool for any computation.\n"
        "Use DuckDuckGoSearchRun for any web/current info.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct."
    )
)


# ----------------------------
# 3) Nodes (reference-style)
# ----------------------------
def assistant_node(state: AgentState, config: RunnableConfig):
    """
    Reference-style assistant node:
    - calls the LLM with [SYSTEM_PROMPT] + state messages
    - returns a new AI message appended into state["messages"]
    """
    response = model.invoke([SYSTEM_PROMPT] + list(state["messages"]), config=config)
    return {"messages": [response]}


# ToolNode handles:
# - reading tool_calls from the last AI message
# - executing tools
# - producing ToolMessage(s)
tool_node = ToolNode(tools)


# ----------------------------
# 4) Graph wiring (reference-style)
# ----------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("assistant")

    # If assistant produced tool_calls -> go to tools, else -> END
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            END: END,  # some versions use END sentinel here
        },
    )

    # After tool execution, go back to assistant
    workflow.add_edge("tools", "assistant")

    return workflow.compile()


# ----------------------------
# 5) Run (same user prompt)
# ----------------------------
def main():
    graph = build_graph()

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then find out about 'Elon Musk's residence' "
        "and then find the latest news at Elon Musk's residence, "
        "and summarize all this in 1 line."
    )

    inputs = {"messages": [HumanMessage(content=query)]}

    out_state = graph.invoke(inputs)
    print(out_state["messages"][-1].content)

    # Optional: stream steps
    # for step in graph.stream(inputs, stream_mode="values"):
    #     msg = step["messages"][-1]
    #     msg.pretty_print()


if __name__ == "__main__":
    main()
