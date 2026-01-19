
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

# Set your API token
# from dotenv import load_dotenv

load_dotenv()   
hf_token = os.getenv("HUGGINGFACE_API_KEY")


# 2. Set task to 'conversational' to satisfy the provider (featherless-ai)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

# 3. Wrap the endpoint in ChatHuggingFace
# This handles the internal chat formatting Gemma-3 requires
chat_model = ChatHuggingFace(llm=llm)

try:
    # 4. Use a Message object instead of a raw string
    messages = [
        HumanMessage(content="What is the capital of India?")
    ]
    
    response = chat_model.invoke(messages)
    print("\n--- AI Response ---\n")
    print(response.content)
    
except Exception as e:
    print(f"Error: {e}")