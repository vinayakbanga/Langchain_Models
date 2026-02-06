# its is used to load content from PDF files and convert eash page 
# into a document that can be processed by language models. It uses the PyPDF2 library to read PDF files and extract text from each page. The extracted text is then stored in a list of documents, which can be further processed or analyzed as needed.

# if there are 25 pages in pdf file then it will create 25 documents and each document will have content of one page. This allows for more granular processing of the PDF content, as each page can be treated as a separate document for tasks such as summarization, question answering, or information extraction.

# limitation: it may not be able to extract text from scanned PDFs or PDFs with complex layouts, as it relies on the text extraction capabilities of the PyPDF2 library. Additionally, it may not preserve the formatting of the original PDF, such as images, tables, or special characters.


import os
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

# 1. Get the absolute path of the directory where THIS script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with your filename to get the full, absolute path
file_path = os.path.join(current_dir, "Vinayak_resume.pdf")

# 3. Pass the full path to the loader

loader=PyPDFLoader(file_path)

docs=loader.load()

print(docs)

print(len(docs))

# to print first page

# print(docs[0].page_content)