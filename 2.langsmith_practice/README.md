# LangSmith Tracing Practice

This folder contains experiments using LangSmith to trace LLM calls with:
- OpenAI models
- Ollama local models

## What this demonstrates
- How to initialise the LangSmith client
- How to trace LLM calls
- How to compare OpenAI vs Ollama behaviour
- How to inspect traces in the LangSmith UI

## Screenshots

### OpenAI Trace
![OpenAI Trace](screenshots/openai_trace.png)

### Ollama Trace
![Ollama Trace](screenshots/ollama_trace.png)

## Example Traces
- OpenAI trace: [OpenAI](https://eu.smith.langchain.com/public/a4afbe99-9ee2-4c26-92e9-a33e3dcc2828/r)
- Ollama trace: [Ollama](https://eu.smith.langchain.com/public/8a8cae4d-fe80-4c08-acf3-4263f279cd2c/r)

## How to run
1. Install dependencies:
   - uv add langchain langchain-openai langchain-ollama langsmith

2. Set environment variables:
   - LANGCHAIN_API_KEY=<your-key>
   - LANGSMITH_TRACING=true
   - LANGCHAIN_PROJECT=<your-project-name>
   - LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com

3. Run:
   - python openai_trace_example.py
   - python ollama_trace_example.py
