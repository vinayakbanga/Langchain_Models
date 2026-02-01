# Runnable passthough is a runnable that returns the input as output without any modification. It is useful in scenarios where you want to maintain the original input data while processing it through a chain of runnables.

# eg where it is used  where i want to see the input along with output in the final result of the chain.

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableParallel
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

joke_generation_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2,model,parser)
})


final_chain= joke_generation_chain | parallel_chain

returned_result=final_chain.invoke({"joke_topic":"computers"})
print("Joke about computers:\n", returned_result["joke"])
print("Explanation of the joke:\n", returned_result["explanation"])
