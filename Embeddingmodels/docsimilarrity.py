
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

token=os.getenv("HUGGINGFACE_API_KEY")

embeddings=HuggingFaceEndpointEmbeddings(model="google/embeddinggemma-300m",huggingfacehub_api_token=token)

documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",    
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",    
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

query = "Which planet is known as the Red Planet?"
# 3. Generate embeddings remotely
try:
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = embeddings.embed_documents(documents)

    # 4. Calculate similarity
    # similarities = cosine_similarity([query_embedding], doc_embeddings)

    # print("Similarities:", similarities)

    
    scores = cosine_similarity([query_embedding], doc_embeddings)[0]

    index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

    print(query)
    print(documents[index])
    print("similarity score is:", score)

    # print(list(enumerate(similarities)))
        
except Exception as e:
    print(f"Inference Error: {e}")
    # print("Ensure you have accepted the Gemma license on Hugging Face.")