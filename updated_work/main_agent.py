import os
import asyncio
import json
from typing import Annotated, Sequence, TypedDict, Any, List, Dict

from rich.console import Console
from rich import print as rich_print

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from tools import setup_tools, build_tool_context
from llm import make_model
from json_utils import _parse_plan
from history_formatters import format_history_for_responder
from nodes import (
    make_planner_node,
    make_validator_node,
    make_executor_node,
    make_gate_node,
    make_responder_node,
)
from routing import route_after_validator, route_after_gate

DEBUG_MESSAGES = 1

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: Dict[str, Any]
    repair_attempts: int
    last_tool_error: Dict[str, Any]

async def build_graph():
    tools = await setup_tools()
    tools_by_name = {t.name: t for t in tools}

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", make_planner_node(tools_by_name))
    workflow.add_node("validator", make_validator_node(tools_by_name))
    workflow.add_node("executor", make_executor_node(tools_by_name))
    workflow.add_node("gate", make_gate_node(tools_by_name, max_repairs=1))
    workflow.add_node("responder", make_responder_node())

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "validator")

    workflow.add_conditional_edges(
        "validator",
        route_after_validator,
        {"executor": "executor", "responder": "responder"},
    )

    workflow.add_edge("executor", "gate")

    workflow.add_conditional_edges(
        "gate",
        route_after_gate,
        {"validator": "validator", "responder": "responder"},
    )

    workflow.add_edge("responder", END)
    return workflow.compile()

async def main():
    graph = await build_graph()

    chat_memory: List[BaseMessage] = []
    console = Console()

    rich_print("\n" + "="*60)
    rich_print("CHARITY AGENT – Advanced Modular Version (Gate + Smart History)")
    rich_print("Type 'exit', 'quit', or 'q' to stop.")
    rich_print("="*60 + "\n")

    while True:
        try:
            user_input = input("User: ")
        except (KeyboardInterrupt, EOFError):
            rich_print("\nGoodbye!")
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            rich_print("\nGoodbye!")
            break

        user_msg = HumanMessage(content=user_input)
        chat_memory.append(user_msg)

        rich_print("\n(Agent is thinking...)\n")

        try:
            async for step in graph.astream(
                {"messages": chat_memory, "repair_attempts": 0},
                stream_mode="updates"
            ):
                for node_name, node_output in step.items():
                    if not node_output:
                        continue
                    if "messages" in node_output:
                        new_messages = node_output["messages"]
                        for msg in new_messages:
                            chat_memory.append(msg)

                            if isinstance(msg, ToolMessage):
                                rich_print(f" ➤ [Tool Executed] {msg.name}")
                            elif isinstance(msg, AIMessage):
                                if getattr(msg, "tool_calls", None) and not (msg.content or "").strip():
                                    continue
                                if "System Note:" not in (msg.content or ""):
                                    rich_print(f"\nAgent: {msg.content}\n")
        except Exception as e:
            rich_print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())