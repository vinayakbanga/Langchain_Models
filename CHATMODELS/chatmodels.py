from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

client = ChatOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

model=ChatOpenAI(model="openai/gpt-4.1-nano")

result=model.invoke("What is the capital of France?")

print(result)

