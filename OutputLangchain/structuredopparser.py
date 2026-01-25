import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
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


schema=[
    ResponseSchema(name="name",description="name of a fictional person"),
    ResponseSchema(name="age",description="age of the fictional person"),
    ResponseSchema(name="city",description="city where the fictional person lives")
    ]

parser=StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    input_variables=[],
    template="Provide a JSON object with keys 'name', 'age', and 'city' describing a fictional person.\n {format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

propmt=template.invoke({})

result=model.invoke(propmt)
parsed_output=parser.parse(result.content)
print("Parsed Output:\n", parsed_output)
print(type(parsed_output))