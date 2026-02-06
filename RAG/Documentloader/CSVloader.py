# Csv loader is used to load content from CSV files and convert each row into a document that can be processed by language models. It uses the pandas library to read CSV files and extract text from each row. The extracted text is then stored in a list of documents, which can be further processed or analyzed as needed.

from langchain_community.document_loaders import CSVLoader

data=CSVLoader(file_path="sample.csv", encoding="utf-8")

print(data.load())

# limitation: it may not be able to handle very large CSV files, as it loads the entire file into memory. Additionally, it may not preserve the formatting of the original CSV, such as special characters or multi-line fields.

