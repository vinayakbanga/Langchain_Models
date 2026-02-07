# My name is vinayak 
# I am learning langchain and RAG.

# I have created a file named RAG.txt where I am noting down all the important points related to RAG.
# I have also created a folder named Textsplitter where I am creating different types of text splitters.

from langchain_text_splitters import RecursiveCharacterTextSplitter

text= """
My name is vinayak 
I am learning langchain and RAG.

I have created a file named RAG.txt where I am noting down all the important points related to RAG.
I have also created a folder named Textsplitter where I am creating different types of text splitters.
"""

spliter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=0)

chunks=spliter.split_text(text)
print(len(chunks))
print(chunks)