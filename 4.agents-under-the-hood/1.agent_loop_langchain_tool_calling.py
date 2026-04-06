from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

#  --- Tools (Langchain @Tools decorator) ---

@tool
def get_product_price(product_name: str) -> float:
    """
    Tool that Look up the price of a product in the catalog.
    Args:
        product_name (str): The name of the product.
    Returns:
        float: The price of the product.
    """
    print(f"    >> Executing get_product_price(product_name='{product_name}')")
    prices = {"laptop": 1299.99, "smartphone": 149.99, "headphones": 199.99}
    return prices.get(product_name, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Tool that applies a discount to a price.
    Args:
        price (float): The original price.
        discount_tier (str): The discount tier to apply.
    Returns:
        float: The discounted price.
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# --- Agent Loop ---

# @traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    # print(tools_dict)

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0) # Initialize the model (without tools) 
    llm_with_tools = llm.bind_tools(tools) # Bind the tools to the model (so that the model can call them) 

    print(f"Question: {question}")
    print("=" * 50)

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
