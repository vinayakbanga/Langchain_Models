import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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


class PersonModel(BaseModel):
    name: str = Field(..., description="name of a fictional person")
    age: int = Field(gt=18, description="age of the fictional person")
    city: str = Field(..., description="city where the fictional person lives")

parser=PydanticOutputParser(pydantic_object=PersonModel)

template= PromptTemplate(
    input_variables=["kind"],
    template="Provide a JSON object with keys 'name', 'age', and 'city' describing a {kind} person.\n {format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt=template.invoke({"kind":"Indian"})

print("Prompt to LLM:\n", prompt)

finalreuslt=parser.parse(model.invoke(prompt).content)

print("Parsed Output:\n", finalreuslt)
print(type(finalreuslt))

