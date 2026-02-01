# Runnable lambda is a simple runnable that allows you to define custom logic using a lambda function. It is useful for quick transformations or computations that do not require a full-fledged class or function definition.

#It acts as a middleare between diff Ai  components,enabling preprocessing,transformation,API calls,filteringa and post processing in langchain workflows.

# from langchain_core.runnables import RunnableLambda

# def word_count(text: str) -> int:
#     return len(text.split())

# word_count_runnable = RunnableLambda(word_count)

# print("Word count of 'Hello world from LangChain':", word_count_runnable.invoke("Hello world from LangChain")) 


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
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


joke_generation_chain=RunnableSequence(prompt,model,parser)

parrallel_chain=RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain= joke_generation_chain | parrallel_chain
returned_result=final_chain.invoke({"joke_topic":"computers"})

print("Joke about computers:\n", returned_result["joke"])
print("Word count of the joke:\n", returned_result["word_count"])


