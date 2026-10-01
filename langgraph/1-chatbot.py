import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages 


class State(TypedDict):
    messages:Annotated[list,add_messages]

model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def chatbot(state:State):
    return {"messages":[model.invoke(state["messages"])]}

graph_builder=StateGraph(State)

graph_builder.add_node("llmChatbot",chatbot)

graph_builder.add_edge(START,"llmChatbot")
graph_builder.add_edge("llmChatbot",END)

graph=graph_builder.compile()

if __name__ == "__main__":
    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        result = graph.invoke({"messages": [("user", user_input)]})
        print(f"Bot: {result['messages'][-1].content}\n")

