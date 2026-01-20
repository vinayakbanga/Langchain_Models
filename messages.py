from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API_KEY")


# 2. Set task to 'conversational' to satisfy the provider (featherless-ai)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

model=ChatHuggingFace(llm=llm)


messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Germany?")
]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)