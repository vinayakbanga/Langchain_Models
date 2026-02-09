# Tools are like hands and legs for LLM agents. They allow agents to interact with the world, perform tasks, and achieve goals. Tools can be anything from a calculator to a web browser, a database, or even another LLM.

# Tool defination : tool is a function or API that is packaged in a way that it can be easily used by an LLM agent. It can be a simple function that takes some input and returns some output, or it can be a complex API that requires authentication and has multiple endpoints.

# Tool usage : To use a tool, an LLM agent needs to know how to call it and what input it requires. This is usually done by providing a description of the tool and its parameters in the prompt. The agent can then generate a call to the tool with the appropriate input.

# 2 type of tool  

# Using duck duck go search engine as an example

# from langchain_community.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun() #tools are runnable

# result = search_tool.invoke("Weather in faridabad")
# print(result)

# using shell tool 
# from langchain_community.tools import ShellTool

# shell_tool = ShellTool()
# result = shell_tool.invoke("echo Hello, World!")
# print(result)

#  How we make custom tools using tool decorator

from langchain_core.tools import tool

# step 1 creare a function that performs the desired task

# def multiply(a,b):
#     """ Multiplies two numbers and returns the result.""" #Docstring is important for the tool to be used by the agent so that llm can understand its a tool and how to use it
#     return a * b

# # Step 2 - add type hints to the function parameters and return value
# def multiply(a: int, b: int) -> int:
#     """ Multiplies two numbers and returns the result."""
#     return a * b

# Step 3 - add tool decorator to the function to make it a tool

# @tool
# def multiply(a: int, b: int) -> int:
#     """ Multiplies two numbers and returns the result."""
#     return a * b

# result = multiply.invoke({"a": 5, "b": 10})
# print(result)

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# Structured tools are tools that have a defined structure for their input and output. 

# https://colab.research.google.com/drive/1GHHGsDFB5266Cc0xDsZ6OWzkB5GGSxFW?usp=sharing

