import os
from langchain_openai import ChatOpenAI

def make_model(temperature: float = 0.0):
    return ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        model="nvidia/nemotron-3-nano-30b-a3b",
        temperature=temperature,
        max_tokens=8192,
    )