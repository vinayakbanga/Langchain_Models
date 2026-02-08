from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

token=os.getenv("HUGGINGFACE_API_KEY")

embeddings=HuggingFaceEndpointEmbeddings(model="google/embeddinggemma-300m",huggingfacehub_api_token=token)



# Create LangChain documents for IPL players

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db', #location where the vector store will be stored
    collection_name='sample'
)

vector_store.add_documents(docs)

# add documents
vector_store.add_documents(docs)

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])

# search documents
vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)

# https://colab.research.google.com/drive/1VwOywJ9LPSIpKWKj9vueVoexSCzGHXNC?usp=sharing#scrollTo=rIKW6J91c2Gw