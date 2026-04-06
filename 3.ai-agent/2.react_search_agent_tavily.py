from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the web for the given query.
#     Args:
#         query (str): The search query.
#     Returns:
#         str: The search results.
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
tools = [TavilySearch()]  # Add the search tool to the list of tools
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the london area on linkedin and list their details?")})
    print(result)

if __name__ == "__main__":
    main()
