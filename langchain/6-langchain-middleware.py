# Middleware is used to intercept and customize what happens between the agent, model, and tools during an agent's execution.
# Modify requests before they reach the LLM.
# Modify responses after the LLM responds.
# Add logging and monitoring.
# Implement guardrails for unsafe or unwanted inputs/outputs.
# Control which tools an agent can use.
# Implement rate limiting or usage controls.
# Add retry/fallback logic when a model or tool fails.
# Dynamically change the model or system prompt.
# Manage agent state and context.
# Add custom business logic without modifying the core agent.

# User
#   ↓
# Agent
#   ↓
# Middleware  ← intercept / modify / control
#   ↓
# Model
#   ↓
# Tool
#   ↓
# Middleware
#   ↓
# Agent
#   ↓
# Response

# Explore BuildIn Middlewares

# 1. Summarization Middleware: It compresses the conversation history into a summary so the agent can continue working without exceeding the model's context window.
    # Summarization based on messages, tokens, fractions


import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

@tool
def search_hotels(city:str)->str:
    """search hotels - returns long response to use more tokens"""
    return f"""Hotels in {city}:
        1. ganrd Hotel
        2. ITC Hotel
        3. Budget hotel"""


agent = create_agent(
    model=model,
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        # based on Message length
        # SummarizationMiddleware(
        #     model=model, trigger=("messages", 10), keep=("messages", 4)
        # ),
        #based on tokens
        SummarizationMiddleware(
                    model=model, trigger=("tokens", 550), keep=("tokens", 200)
                )
        
    ],
)

config = {"configurable": {"thread_id": "test-1"}}

questions = [
    "what is 2+2?",
    "what is 2*2?",
    "what is 50%2?",
    "what is 15-7?",
    "what is 3*3?",
    "what is 4*4?",
    "what is 2+2?",
    "what is 2*2?",
    "what is 50%2?",
    "what is 15-7?",
    "what is 3*3?",
    "what is 4*4?",
]

# for q in questions:
#     response = agent.invoke({"messages": [HumanMessage(content=q)]}, config)
#     print(f"Messages: {response}")
#     print(f"Messages: {len(response["messages"])}")

#Token counter
def count_tokens(messages):
    total_chars=sum(len(str(m.content)) for m in messages)
    return total_chars // 4

# Run test
cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

# for city in cities:
#     response = agent.invoke(
#         {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
#         config=config
#     )

#     tokens = count_tokens(response["messages"])
#     print(f"{city}: {tokens} tokens, {len(response['messages'])} messages")
#     print(f"{(response['messages'])}")


#=============================================
# Human in loop middleware
# check on ur own in docs
