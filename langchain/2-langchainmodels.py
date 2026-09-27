### Models
### here we can use chatOpenAI, ChatGoogleGenerativeAI, init_chat_model

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-3.8-flash")
response = model.invoke("hello how are you?")
print(response.content)

## Streaming
for chunk in model.stream("write a details explaination of how llm works"):
    print(chunk)

## Batching
responses= model.batch(["where do llms get stored","why am i learning langchain","what is the need to learn GenAI"])
for response in responses:
    print(response)

