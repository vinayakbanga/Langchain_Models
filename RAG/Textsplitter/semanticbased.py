

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker

import os

load_dotenv()

token=os.getenv("HUGGINGFACE_API_KEY")

embeddings=HuggingFaceEndpointEmbeddings(model="google/embeddinggemma-300m",huggingfacehub_api_token=token)


sample_text = """
Farmers were working hard in the fields, preparing the soil and planting seeds for 
the next season. The sun was bright, and the air smelled of earth and fresh grass. 
The Indian Premier League (IPL) is the biggest cricket league in the world. People 
all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates 
fear in cities and villages. When such attacks happen, they leave behind pain and 
sadness. To fight terrorism, we need strong laws, alert security forces, and support 
from people who care about peace and safety.
"""

text_splitter=SemanticChunker(
    embeddings,breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5
)

docs=text_splitter.create_documents([sample_text])

print(len(docs))

print(docs)