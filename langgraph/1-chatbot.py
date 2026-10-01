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

# if __name__ == "__main__":
#     print("Chatbot ready. Type 'quit' to exit.\n")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ("quit", "exit", "q"):
#             break
#         result = graph.invoke({"messages": [("user", user_input)]})
#         print(f"Bot: {result['messages'][-1].content}\n")


#===========================================
#Tool calling with LangGraph

from langchain_tavily import TavilySearch

tavilytool=TavilySearch(max_results=2)

def multiply(a:int,b:int)-> int:
    """ Multiply the a and b"""
    return a*b

tools=[tavilytool,multiply]
llm_with_tools=model.bind_tools(tools)

from langgraph.prebuilt import ToolNode, tools_condition

#node defination
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

#Graph
builder=StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

#Add Edges
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm",tools_condition)
builder.add_edge("tools",END)

graph= builder.compile()

# res=graph.invoke({"messages":"what is recent ai news"})
# res=graph.invoke({"messages":"what is 2 multiple by 3"})
# print(res)

#===========================================
#Tool calling with LangGraph with ReAct Architecture and adding memroy

from langchain_tavily import TavilySearch

tavilytool=TavilySearch(max_results=2)

def multiply(a:int,b:int)-> int:
    """ Multiply the a and b"""
    return a*b

tools=[tavilytool,multiply]
llm_with_tools=model.bind_tools(tools)

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
memory=MemorySaver()

#node defination
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

#Graph
builder=StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

#Add Edges
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm",tools_condition)
builder.add_edge("tools","tool_calling_llm")

graph= builder.compile(checkpointer=memory)
config={"configurable":{"thread_id":"1"}}

# res=graph.invoke({"messages":"what is recent ai news"})
# res=graph.invoke({"messages":"what is recent ai news and what is 2 multiple by 3"})
# res=graph.invoke({"messages":"Hi, my name is eshwar"},config=config)
res=graph.invoke({"messages":"what is my name"},config=config)
for m in res["messages"]:
    print(m.pretty_print())