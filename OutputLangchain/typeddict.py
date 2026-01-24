from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API_KEY")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

model=ChatHuggingFace(llm=llm)

class ReviewRequest(TypedDict):
    review_text: str
    sentiment: str

structured_model=model.with_structured_output(ReviewRequest)

result=structured_model.invoke("this phone is great and I love it")

# print(result)
print("Review Text:", result['review_text'])
print("Sentiment:", result['sentiment'])