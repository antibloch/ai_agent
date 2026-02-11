import os
from typing import Any, Dict, List, Tuple

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import BaseMessage



def message_to_text(msg: BaseMessage) -> str:
    """Return a clean string from LangChain message content across providers."""
    c: Any = getattr(msg, "content", "")

    # Most providers: plain string
    if isinstance(c, str):
        return c.strip()

    # Gemini / some others: list of parts like [{"type":"text","text":"..."}]
    if isinstance(c, list):
        chunks = []
        for part in c:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                # Most common: {"type":"text","text":"..."}
                if "text" in part and isinstance(part["text"], str):
                    chunks.append(part["text"])
                # Fallback: stringify unknown part
                else:
                    chunks.append(str(part))
            else:
                chunks.append(str(part))
        return "\n".join(x.strip() for x in chunks if x and x.strip())

    # Fallback
    return str(c).strip()


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

    # model = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0.0,
    # )

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # example; pick what you have access to
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
        system_prompt=system_prompt,
        debug=True,
        name="python_search_agent",
        )


    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'Elon Musk's net worth' "
        "and summarize in 1 line."
    )

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # result is an AgentState-like dict; final messages contain the answer
    # final_msg = result["messages"][-1]
    # print(final_msg.content)

    final_msg = result["messages"][-1]
    print(message_to_text(final_msg))


if __name__ == "__main__":
    main()
