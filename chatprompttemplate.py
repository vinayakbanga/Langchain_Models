from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# in prompt temaplate the prompt can be made like this dynammicly
# chat_template = ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {domain} assistant."),
#     HumanMessage(content="Please explain the concept of {topic} "),
# ])

# prompt=chat_template.invoke({
#     "domain": "Computer Science",
#     "topic": "Cloud Computing"
# })

chat_template = ChatPromptTemplate([
    ('system',"You are a helpful {domain} assistant."),
    ('human',"Please explain the concept of {topic} "),
    
])

prompt=chat_template.invoke({
    "domain": "Computer Science",
    "topic": "Cloud Computing"
})


print(prompt)