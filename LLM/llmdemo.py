import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

load_dotenv()

# # First API call with reasoning
# response = client.chat.completions.create(
#   model="openai/gpt-oss-120b:free",
#   messages=[
#           {
#             "role": "user",
#             "content": "How many r's are in the word 'strawberry'?"
#           }
#         ],
#   extra_body={"reasoning": {"enabled": True}}
# )

# # Extract the assistant message with reasoning_details
# response = response.choices[0].message

llm= OpenAI(model="openai/gpt-4.1-nano")

response=llm.invoke("What is the capital of France?")

print(response)