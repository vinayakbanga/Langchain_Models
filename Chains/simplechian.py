import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Set your API token
# from dotenv import load_dotenv

load_dotenv()   
hf_token = os.getenv("HUGGINGFACE_API_KEY")
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

# 3. Wrap the endpoint in ChatHuggingFace
# This handles the internal chat formatting Gemma-3 requires
model = ChatHuggingFace(llm=llm)


prompt= PromptTemplate(
    template="Generate 5 intrecting facts about {topic} in 3 points",
    input_variables=["topic"]
)

# for chains string outputparser is used

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({"topic":"Iphone"})
  






   
print("Final Result:\n", result)


# chain.get_graph().print_ascii()