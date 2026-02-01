import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
# Set your API token
# from dotenv import load_dotenv

load_dotenv()   
hf_token = os.getenv("HUGGINGFACE_API_KEY")
llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

llm2= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)
# 3. Wrap the endpoint in ChatHuggingFace
# This handles the internal chat formatting Gemma-3 requires
model1 = ChatHuggingFace(llm=llm1)
model2=ChatHuggingFace(llm=llm2)

prompt1= PromptTemplate(
    template="Generate a tweet about {topic}  in 3 lines",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="Generate a linkedin post on\n{topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    "tweet": prompt1 | model1 | parser,   #this is another way of denoting runnable sequence
    "linkedin_post": prompt2 | model2 | parser,

})

chain= parallel_chain
result=chain.invoke({"topic":"Artificial Intelligence"})

# print("Final Result from Parallel Runnable:\n", result)

print("\nTweet about Artificial Intelligence:\n", result["tweet"])
print("\nLinkedin Post about Artificial Intelligence:\n", result["linkedin_post"])