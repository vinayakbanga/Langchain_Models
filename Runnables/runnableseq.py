from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv

load_dotenv()   
hf_token = os.getenv("HUGGINGFACE_API_KEY")
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

prompt= PromptTemplate(
    input_variables=["joke_topic"],
    template="Tell me a joke about {joke_topic}."
)


parser=StrOutputParser()

prompt2=PromptTemplate(
    input_variables=["joke"],
    template="explain the joke:\n{joke}"
)



chain=RunnableSequence(prompt,model,parser,prompt2,model,parser)

returned_result=chain.invoke({"joke_topic":"computers"})

print("Joke about computers:\n", returned_result)
