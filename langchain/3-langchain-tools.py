#   Models can request to call tools that perform tasks such as fetching data from a database, searching the web or running code.
#   Tools are pairings of:
#    1. A schema, including the name of tool, a description, and/or argument definitions (often a JSON schema)
#    2. A function or coroutine to execute

import os
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

@tool
def get_weather(location:str)-> str:
    """Get weather at a location"""
    return f"it's sunny in {location}"

model_with_tools = model.bind_tools([get_weather])
# response = model_with_tools.invoke("what is the waether in dallas?")
# print(response)
# for tool_call in response.tool_calls:
#     print(f"Tool:{tool_call['name']}")
#     print(f"Args:{tool_call['args']}")


#Tool Execution Loops
#step 1: Model generates tool call
messages=[{"role":"user","content":"what's the weather in Boston?"}]
ai_msg= model_with_tools.invoke(messages)
messages.append(ai_msg)

# step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

#step 3: pass results back to model for final response
final_response=model_with_tools.invoke(messages)
print(final_response.text)
print(messages)
