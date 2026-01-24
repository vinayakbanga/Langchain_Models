from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# chat trmplate

chat_template = ChatPromptTemplate([
    ('system',"You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human',"Please assist the customer with their issue: {issue} "),
])

chat_history = []

# Load chat history

with open("chat_history.txt") as file:
    chat_history.extend(file.readlines())

print(chat_history)

# creating prompot

prompt=chat_template.invoke({'chat_history': chat_history, 'issue': 'Unable to access my account.'})
print(prompt)