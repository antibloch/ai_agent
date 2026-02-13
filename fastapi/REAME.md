# FastAPI Server
```

export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qc:latest"
uvicorn app:app --host 0.0.0.0 --port 8000



curl -s http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compute pi upto 13 decimal places using python, then search the web for latest Elon Musk net worth and summarize in 1 line.",
    "include_trace": true
  }'


```

# Nodejs api

```
node server.js

node server.js --trace

```