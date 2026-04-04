# LangChain Search Agents

This folder demonstrates how to build search agents using LangChain's `create_agent` interface and progresses through three key concepts, showing how to evolve from a basic custom tool implementation to using structured outputs with built-in LangChain integrations.

## 🧠 What is an AI Agent?

An AI Agent is an LLM that can **take actions**, not just generate text.  
It can think about a problem, decide the next step, use tools, observe results, and continue working until it completes the task.

In simple terms:

**An AI Agent is an LLM that can reason step‑by‑step, use external tools or APIs, and work toward a goal instead of giving a single response.**

## Learning Objectives

- Understand the LangChain `create_agent` interface
- Build custom tools using the `@tool` decorator
- Integrate third-party search APIs (Tavily)
- Use LangChain's built-in tool integrations
- Implement structured outputs with Pydantic models

## How to run
1. Install dependencies:
   ```bash
   pip install tavily-python
   ```

2. Set environment variables:
   ```bash
   # Create a .env file with:
   export OPENAI_API_KEY=your_openai_key
   export TAVILY_API_KEY=your_tavily_key
   ```

3. Run:
   ```bash
   python react_search_agent.py
   ```

## Example Query

The agent searches for AI engineer job postings in the London Area on LinkedIn:
```python
"search for 3 job postings for an ai engineer using langchain in the London area on linkedin and list their details?"
```

## Key Takeaways

- **Custom Tools**: You can create custom tools using the `@tool` decorator for specialized functionality
- **Built-in Integrations**: LangChain provides pre-built tools that reduce boilerplate and improve maintainability
- **Structured Outputs**: Using Pydantic models with `response_format` ensures type-safe, predictable agent responses
- **Agent Interface**: The `create_agent` function provides a simple, consistent interface for building agents with different capabilities
