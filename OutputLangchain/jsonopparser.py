import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

parser=JsonOutputParser()
template= PromptTemplate(
    input_variables=[],
    template="Provide a JSON object with keys 'name', 'age', and 'city' describing a fictional person.\n {format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()},

)

# prompt=template.format()

# print("Prompt to LLM:\n", prompt)

# result=model.invoke(prompt)

# parsed_output=parser.parse(result.content)
# in place of line 31 and 35,37 we can also write like

chain=template | model | parser


# print("Raw LLM Output:\n", result.content)

parsed_output=chain.invoke({})

print("Parsed Output:\n", parsed_output)
print(type(parsed_output))
print(parsed_output['name'])



