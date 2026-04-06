from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
import json

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
    Available tiers: bronze, silver, gold
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

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    # print(tools_dict)

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0) # Initialize the model (without tools) 
    llm_with_tools = llm.bind_tools(tools) # Bind the tools to the model (so that the model can call them) 

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # ai_message = llm_with_tools.invoke(messages)

    # tools_called = ai_message.tool_calls

    # # print(f"\n--- Agent Response ---")
    # # print the AI response in json format to show the tool calls clearly
    # # print(ai_message.model_dump_json(indent=2))
    # print(f"AI: {ai_message.content}\n")
    # print(f"Tools called: {tools_called[0]}\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        ai_message = llm_with_tools.invoke(messages)

        tools_called = ai_message.tool_calls

        print(f"    AI: {ai_message.content}\n")
        print(f"    Tools called: {tools_called}\n")

        if not tools_called:
            print("No tools called. Assuming the agent has arrived at a final answer.")
            return ai_message.content
        
        # Process only the first tool call in this iteration (if multiple tools are called, they will be processed in subsequent iterations)
        tool_call = tools_called[0] # get the first tool call
        tool_name = tool_call.get("name") # get the name of the tool called
        tool_args = tool_call.get("args", {}) # get the arguments passed to the tool (default to empty dict if no args)
        tool_call_id = tool_call.get("id") # get the unique id of the tool call (to match it with the observation later)

        print(f"    Processing tool call: {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name) # get the actual tool function to call based on the tool name
        if tool_to_use is None: # if the tool name is not found in our tools dict, raise an error (this should not happen if the model is calling tools we have provided, but it's good to check)
            raise ValueError(f"Error: Tool '{tool_name}' not found.")
         
        observation = tool_to_use.invoke(tool_args) # call the tool with the provided arguments and get the observation (result) - using .invoke() to call the tool function with the arguments as a dict
        # observation = tool_to_use(tool_args)
        print(f"    Observation from tool '{tool_name}': {observation}")

        # Add the tool call and its observation to the messages for the next iteration
        # ai_message is appending, So the agent remembers what tool it asked for.
        messages.append(ai_message) # add the AI response that included the tool call
        # ToolMessage is appending, So the agent sees the tool response and can continue reasoning.
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Max iterations reached without arriving at a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
