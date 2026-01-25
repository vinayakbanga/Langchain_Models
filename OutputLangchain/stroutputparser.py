import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
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

#1st prompt -> detailed explanation of research paper
template= PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="Explain the research paper titled '{paper}' in a '{style}' manner with a '{length}' length.",
)

#2nd promt-> summarize the detailed explanation in bullet points
template2= PromptTemplate(
    input_variables=["detailed_explanation"],
    template="Summarize the following detailed explanation into concise bullet points:\n{detailed_explanation}",
)

prompt1=template.invoke({
    "paper": "Attention Is All You Need",
    "style": "Beginner-Friendly",
    "length": "Short (1-2 paragraphs)"})

result1=model.invoke(prompt1)

prompt2=template2.invoke({
    "detailed_explanation": result1.content})

result2=model.invoke(prompt2)
print("Detailed Explanation:\n", result1.content)



# this is with prompt template