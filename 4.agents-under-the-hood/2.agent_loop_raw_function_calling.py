from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable
# import json

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

#  --- Tools (Langchain @Tools decorator) ---

@traceable(run_type="tool")
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

@traceable(run_type="tool")
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

# Difference : Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Tool that Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Tool that applies a discount to a price. Available tiers: bronze, silver, gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
# However, this requires your docstrings to follow the Google docstring format
# tools_for_llm = [get_product_price, apply_discount]

# --- Helper: traced Ollama call ---
# Difference : Without LangChain, we must manually trace LLM calls for LangSmith.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
# --- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }
    # print(tools_dict)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Difference : ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tools_called = ai_message.tool_calls

        print(f"    AI: {ai_message.content}\n")
        print(f"    Tools called: {tools_called}\n")

        if not tools_called:
            print("No tools called. Assuming the agent has arrived at a final answer.")
            return ai_message.content
        
        # Process only the first tool call in this iteration (if multiple tools are called, they will be processed in subsequent iterations)
        tool_call = tools_called[0] # get the first tool call
        # Difference : Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    Processing tool call: {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name) # get the actual tool function to call based on the tool name
        if tool_to_use is None: # if the tool name is not found in our tools dict, raise an error (this should not happen if the model is calling tools we have provided, but it's good to check)
            raise ValueError(f"Error: Tool '{tool_name}' not found.")
         
        # Difference : Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args) # call the tool with the provided arguments and get the observation (result) - using .invoke() to call the tool function with the arguments as a dict
        # observation = tool_to_use(tool_args)
        print(f"    Observation from tool '{tool_name}': {observation}")

        # Add the tool call and its observation to the messages for the next iteration
        # ai_message is appending, So the agent remembers what tool it asked for.
        messages.append(ai_message) # add the AI response that included the tool call
        # ToolMessage is appending, So the agent sees the tool response and can continue reasoning.
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Max iterations reached without arriving at a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
