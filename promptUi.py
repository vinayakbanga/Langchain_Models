import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_core.messages import HumanMessage
import streamlit as st
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


st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="Explain the research paper titled '{paper}' in a '{style}' manner with a '{length}' length.",
)

# Fill place holdesrs

prompt=template.invoke({
    "paper": paper_input,
    "style": style_input,
    "length": length_input})


if st.button("Generate Response"):
    result=model.invoke(prompt)
    st.write(prompt)
    st.write(result.content)
    # st.text("Generating response...")
