import os
from langchain_community.document_loaders import TextLoader

# 1. Get the absolute path of the directory where THIS script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with your filename to get the full, absolute path
file_path = os.path.join(current_dir, "textloadersample.txt")

# 3. Pass the full path to the loader
try:
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    print("Success! Document loaded.")
    print(docs[0])
except Exception as e:
    print(f"Still failing: {e}")


    # can give this doc to llm also like

    # chain= prompt | model | parser

# print(chain.invoke(docs[0].page_content))
