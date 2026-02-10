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
```