from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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

chat_history=[
    SystemMessage(content="You are a helpful assistant.")

]


while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    response=model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))   
    print("AI:", response.content)

print(chat_history)