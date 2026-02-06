# Webbase loader is used to load content from web pages. It uses the BeautifulSoup library to parse the HTML content of the web page and extract the text. The extracted text is then stored in a document that can be processed by language models.

# when to use:
# Webbase loader is useful when you want to extract information from web pages, such as news articles, blog posts, or any other type of content that is available on the web. It can be used for tasks such as summarization, question answering, or information extraction from web pages.

# limitation: Webbase loader may not be able to extract text from web pages that have complex layouts, dynamic content, or require authentication. Additionally, it may not preserve the formatting of the original web page, such as images, tables, or special characters.

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()   
hf_token = os.getenv("HUGGINGFACE_API_KEY")
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)


url="https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421?pid=COMH64PY76CJKBYU&lid=LSTCOMH64PY76CJKBYUOL7TOK&marketplace=FLIPKART&store=6bo%2Fb5g&spotlightTagId=default_BestsellerId_6bo%2Fb5g&srno=b_1_4&otracker=browse&fm=organic&iid=25e941ed-1aea-4d70-9396-4835c5a98a84.COMH64PY76CJKBYU.SEARCH&ppt=browse&ppn=browse&ssid=u3ndh4txhc0000001770403257232"

loader=WebBaseLoader(url)

docs=loader.load()

prompt= PromptTemplate(
    input_variables=["product_details"],
    template="give me a specification for this product:\n{product_details}"
)

parser=StrOutputParser()

chain= prompt | model | parser

# print(len(docs))

# print(docs)

returned_result=chain.invoke({"product_details":docs[0].page_content})

print("Product specification:\n", returned_result)