### Langchain version v1.4.2 (fixed for LangGraph)
### Agents
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
import langchain
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

def get_weather(city:str)->str:
    """Get weather for a city"""
    return f"The weather in {city} is sunny."

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is weather in bangalore?"}
    ]
})

print(response["messages"][-1].content)