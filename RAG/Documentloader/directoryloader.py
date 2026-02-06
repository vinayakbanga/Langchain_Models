# Directory loaders are used to load documents from a directory. They can be used to load documents in various formats, such as PDF, Word, and text files. The DirectoryLoader class is a base class for loading documents from a directory, and it can be extended to support different file formats.

# in this example, we are using the DirectoryLoader class to load PDF files from a directory. We specify the path to the directory, the glob pattern to match PDF files, and the loader class to use for loading the PDF files. The loader will then load all the PDF files in the specified directory and return a list of documents that can be processed by language models.

# lazy_loading is a technique where documents are loaded on demand, rather than all at once. This can be useful when dealing with large directories of documents, as it can help to reduce memory usage and improve performance. With lazy loading, documents are only loaded when they are needed, which can help to speed up the loading process and reduce the overall time it takes to load the documents.


import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

# 1. Get the absolute path of the directory where THIS script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with your filename to get the full, absolute path
file_path = os.path.join(current_dir, "dirloadereg")



loader=DirectoryLoader(
    path=file_path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)




# docs=loader.load()
docs= loader.lazy_load()


# print(len(docs))
# print(docs[0].page_content)

for doc in docs:
    print(doc.metadata)


