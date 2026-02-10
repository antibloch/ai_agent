import os
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

# NEW: partner package (replaces deprecated langchain_community.llms.HuggingFacePipeline)
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

os.environ["USER_AGENT"] = "my-langchain-agent/1.0"


# ---------------------------------------------------------------------
# Build local HuggingFace Chat Model (Qwen)
# ---------------------------------------------------------------------
def build_hf_chat_model(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 512,
):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,           # greedy
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        # temperature omitted because do_sample=False
    )

    # Wrap HF pipeline in LangChain LLM adapter, then in Chat adapter
    llm = HuggingFacePipeline(pipeline=gen_pipe)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


# ---------------------------------------------------------------------
# Tools: Python + Internet Search
# ---------------------------------------------------------------------
def build_tools():
    search = DuckDuckGoSearchRun()
    python = PythonREPLTool()
    return [search, python]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    tools = build_tools()
    model = build_hf_chat_model()

    system_prompt = (
        "You are a tool-using AI assistant.\n"
        "Use PythonREPLTool for any computation.\n"
        "Use DuckDuckGoSearchRun for any web or current information.\n"
        "If no tool is needed, answer directly.\n"
        "Be concise and correct."
    )

    # FIX: prompt -> system_prompt (LangChain v1 API)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    query = (
        "Compute pi upto 13 decimal places using python, "
        "then search the web for 'Elon Musk's net worth' "
        "and summarize in 1 line."
    )

    result: Dict[str, Any] = agent.invoke({"messages": [HumanMessage(content=query)]})

    print("\nFinal Answer:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
