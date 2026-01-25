import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

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

class FeedbackSchema(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
         description="The sentiment of the text"
    )

parser2=PydanticOutputParser(pydantic_object=FeedbackSchema)

prompt1= PromptTemplate(
    template="Classify the following text into Positive or Negative\n{feedback}/n{format_instructions}",
    input_variables=["feedback"]
    ,partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain=prompt1 | model | parser2

# classification=classifier_chain.invoke({"text":"THis is the worst smartphone."}).sentiment

# print("Sentiment Classification:\n", classification)

prompt2= PromptTemplate(
    template="Write an appropiate response to this positive feedback{feedback}",
    input_variables=["feedback"]   
)

prompt3= PromptTemplate(
    template="Write an appropiate response to this negative feedback{feedback}",    
    input_variables=["feedback"]
)

brach_chain=RunnableBranch(
    (lambda x:x.sentiment=="Positive", prompt2 | model | parser),
    (lambda x:x.sentiment=="Negative", prompt3 | model | parser),
    RunnableLambda(lambda x: Exception("Invalid Sentiment"))
)

chain=classifier_chain | brach_chain

result=chain.invoke({"feedback":"I hate the new features of this smartphone. They are so useless!"})

print("Final Result:\n", result)
chain.get_graph().print_ascii()