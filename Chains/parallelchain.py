import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
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
    template="Generate short and simple notes on {topic}  in 5 lines",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="Generate 5 questions on\n{topic}",
    input_variables=["topic"]
)

prompt3= PromptTemplate(
    template="Merge the following notes and quix into a single document:\n Notes:{notes} and Questions:{questions}",
    input_variables=["notes","questions"]
)

parser=StrOutputParser()

parallelchain=RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "questions": prompt2 | model2 | parser,

})

merge_chain=prompt3 | model1 | parser

chain= parallelchain | merge_chain


result=chain.invoke({"topic":"Cloud Computing"})


print("Final Result:\n", result)

chain.get_graph().print_ascii()


