# Conda environment setup
```
# Create a new conda env with Python 3.10
conda create -n langchain_agents python=3.10 -y

# Activate it
conda activate langchain_agents

pip install -U langchain langchain-community langchain-experimental langgraph langchain-openai duckduckgo-search torch torchvision torchaudio transformers accelerate langchain-huggingface huggingface-hub

pip install langchain-ollama
pip install -U ddgs

pip install google-search-results

pip install pyowm

pip install -U langchain-google-genai

pip install geopy

pip install -U mcp langchain-mcp-adapters

pip install pymongo

pip install maxim-py

pip install -U langsmith

pip install "fastapi[all]"

pip install "uvicorn[standard]"

```


# LangSmith monitoring
```
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY="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"
export LANGSMITH_PROJECT="ollama-react-dev"

# Strongly recommended for short scripts so traces flush before exit:
export LANGCHAIN_CALLBACKS_BACKGROUND=false
```