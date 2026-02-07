# Length based text spliiting - Length based text splitting is a method of dividing a large text into smaller segments based on a specified character or token limit,
# or word count. This technique is useful for processing and analyzing large documents, as it allows for easier handling 
# and manipulation of the text. The length-based approach can be particularly beneficial when working with language models, 
# as it helps to ensure that the input text does not exceed the model's maximum token limit.
# However, it may not always preserve the natural structure of the text, such as sentences or paragraphs, which can affect the coherence and meaning of the segments.

# Advantages: It is simple to implement and can be effective for processing large texts that do not require preserving the original structure. It can also help to manage memory usage when working with large documents.
# it is fast and efficient, as it does not require complex algorithms or analysis of the text structure. It can be easily customized to fit specific requirements by adjusting the character or token limit.
# Disadvantages: It may result in segments that are not coherent or meaningful, as it does not take into account the natural structure of the text. It may also lead to loss of context if important information is split across segments.

# from langchain_text_splitters import CharacterTextSplitter

# text="Length based text spliiting - Length based text splitting is a method of dividing a large text into smaller segments based on a specified character or token limit,or word count. This technique is useful for processing and analyzing large documents, as it allows for easier handling and manipulation of the text. The length-based approach can be particularly beneficial when working with language models, as it helps to ensure that the input text does not exceed the model's maximum token limit.However, it may not always preserve the natural structure of the text, such as sentences or paragraphs, which can affect the coherence and meaning of the segments.Advantages: It is simple to implement and can be effective for processing large texts that do not require preserving the original structure. It can also help to manage memory usage when working with large documents.it is fast and efficient, as it does not require complex algorithms or analysis of the text structure. It can be easily customized to fit specific requirements by adjusting the character or token limit.Disadvantages: It may result in segments that are not coherent or meaningful, as it does not take into account the natural structure of the text. It may also lead to loss of context if important information is split across segments."


# splitter=CharacterTextSplitter(separator="\n",chunk_size=10,chunk_overlap=0)

# result=splitter.split_text(text)

# print(result)

# using doc lader and text splitter together

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import os

# 1. Get the absolute path of the directory where THIS script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with your filename to get the full, absolute path
file_path = os.path.join(current_dir, "Vinayak_Resume.pdf")

# 3. Pass the full path to the loader

loader=PyPDFLoader(file_path)

docs=loader.load()

splitter=CharacterTextSplitter(separator="\n",chunk_size=200,chunk_overlap=0)

result=splitter.split_documents(docs)

print(result[0].page_content)


