# ref: https://ai.google.dev/gemini-api/docs/langgraph-example

import os
import json
from datetime import date
from typing import Any, Annotated, Sequence, TypedDict

import requests
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]  # or set directly

# ----------------------------
# Helpers
# ----------------------------
def message_to_text(msg: BaseMessage) -> str:
    """Return a clean string from LangChain message content across providers."""
    c: Any = getattr(msg, "content", "")

    if isinstance(c, str):
        return c.strip()

    if isinstance(c, list):
        chunks = []
        for part in c:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if "text" in part and isinstance(part["text"], str):
                    chunks.append(part["text"])
                else:
                    chunks.append(str(part))
            else:
                chunks.append(str(part))
        return "\n".join(x.strip() for x in chunks if x and x.strip())

    return str(c).strip()


def _normalize_tool_args(args: Any):
    """
    Tool-call args may arrive as:
      - dict with "__arg1" (common for single-input tools)
      - dict with named fields (args_schema tools)
      - raw string
    Normalize so `.invoke(...)` works reliably.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        if "__arg1" in args:
            return args["__arg1"]
        return args
    return args


# ----------------------------
# Environment
# ----------------------------
os.environ["USER_AGENT"] = "my-langchain-agent/1.0"
# IMPORTANT: set GEMINI_API_KEY in your environment
# export GEMINI_API_KEY="..."


# ----------------------------
# 1) Graph state
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int


# ----------------------------
# 2) Tools
# ----------------------------
# Weather tool (from your reference, slightly adjusted so ToolMessage content is always a string)
geolocator = Nominatim(user_agent="weather-app")


class WeatherInput(BaseModel):
    location: str = Field(description="The city and state, e.g., San Francisco")
    date: str = Field(description="Forecast date in yyyy-mm-dd")


@tool("get_weather_forecast", args_schema=WeatherInput)
def get_weather_forecast(location: str, date: str) -> str:
    """
    Retrieves hourly temperature for a given location + date using Open-Meteo.
    Returns JSON text (string) so it safely fits ToolMessage.content.
    """
    loc = geolocator.geocode(location)
    if not loc:
        return json.dumps({"error": "Location not found"}, ensure_ascii=False)

    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={loc.latitude}&longitude={loc.longitude}"
            f"&hourly=temperature_2m&start_date={date}&end_date={date}"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        times = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        out = {t: temp for t, temp in zip(times, temps)}
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def build_tools():
    return [
        DuckDuckGoSearchRun(),  # single-input string tool
        PythonREPLTool(),       # single-input string tool
        get_weather_forecast,   # args_schema tool (dict)
    ]


tools = build_tools()
tools_by_name = {t.name: t for t in tools}


# ----------------------------
# 3) Model (Gemini) + bind tools
# ----------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in environment.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    max_retries=2,
    google_api_key=api_key,
)

model = llm.bind_tools(tools)  # CRITICAL for ReAct tool_calls


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a tool-using assistant.\n"
        "Use PythonREPLTool for computations.\n"
        "Use DuckDuckGoSearchRun for web/current events.\n"
        "Use get_weather_forecast for weather.\n"
        "If you did not call a tool, DO NOT claim you did.\n"
        "Be concise and correct."
    )
)


# ----------------------------
# 4) Nodes
# ----------------------------
def call_tool(state: AgentState):
    outputs = []
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    for tc in tool_calls:
        name = tc["name"]
        tool_obj = tools_by_name.get(name)

        if tool_obj is None:
            outputs.append(
                ToolMessage(
                    content=f"Unknown tool: {name}",
                    name=name,
                    tool_call_id=tc.get("id", ""),
                )
            )
            continue

        args = _normalize_tool_args(tc.get("args"))
        try:
            result = tool_obj.invoke(args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
        except Exception as e:
            result = f"Tool error: {type(e).__name__}: {e}"

        outputs.append(
            ToolMessage(
                content=result,
                name=name,
                tool_call_id=tc.get("id", ""),
            )
        )

    return {
        "messages": outputs,
        "number_of_steps": state["number_of_steps"] + 1,
    }


def call_model(state: AgentState, config: RunnableConfig):
    response = model.invoke([SYSTEM_PROMPT] + list(state["messages"]), config=config)
    return {
        "messages": [response],
        "number_of_steps": state["number_of_steps"] + 1,
    }


# ----------------------------
# 5) Edge logic
# ----------------------------
def should_continue(state: AgentState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "continue" if tool_calls else "end"


# ----------------------------
# 6) Build graph
# ----------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)

    workflow.set_entry_point("llm")

    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {"continue": "tools", "end": END},
    )

    workflow.add_edge("tools", "llm")

    return workflow.compile()


# ----------------------------
# 7) Run
# ----------------------------
def main():
    graph = build_graph()

    today = date.today().isoformat()
    query = (
        "1) Compute pi up to 13 decimal places using python.\n"
        "2) Then search the web for 'Elon Musk net worth latest' and summarize in 1 line.\n"
        f"3) Also tell me the weather in Berlin on {today} using the weather tool.\n"
    )

    inputs: AgentState = {
        "messages": [HumanMessage(content=query)],
        "number_of_steps": 0,
    }

    # If you want to SEE the ReAct loop, stream:
    # for st in graph.stream(inputs, stream_mode="values"):
    #     last_msg = st["messages"][-1]
    #     # pretty_print() is nice, but Gemini content can be multipart; print safe text:
    #     print("\n--- step", st["number_of_steps"], "---")
    #     print(message_to_text(last_msg))

    # Or just do a single invoke and print final:
    out = graph.invoke(inputs)
    print(message_to_text(out["messages"][-1]))


if __name__ == "__main__":
    main()
