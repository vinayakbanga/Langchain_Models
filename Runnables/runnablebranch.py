#Runnable branch is used to create conditional logic within a chain of runnables. It allows you to define different branches of execution based on certain conditions or criteria.

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda,RunnablePassthrough


# Set your API token
# from dotenv import load_dotenv

load_dotenv()   
hf_token = os.getenv("HUGGINGFACE_API_KEY")
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="conversational", 
    # max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt=PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]   
)

prompt2=PromptTemplate(
    template="Write a brief summary on {text}.",
    input_variables=["text"]
)

report_gen_chian=prompt | model | parser

branch_chain=RunnableBranch(
    # 'x' is the string output from the report_gen_chain
(lambda x: len(x) > 500, (lambda x: {"text": x}) | prompt2 | model | parser),
    RunnablePassthrough(),
    # RunnableLambda(lambda x: Exception("Invalid Input"))
)

final_chain= report_gen_chian | branch_chain

printed_result=final_chain.invoke({"topic":"Artificial Intelligence advancements in the year 2024 focusing on various sectors including healthcare, finance, education, and transportation. The report should cover the latest research, applications, ethical considerations, and future trends in AI technology. Additionally, it should highlight key companies and startups that are leading innovation in this field, as well as government policies and regulations impacting AI development globally."})

print("Final Result:\n", printed_result)




