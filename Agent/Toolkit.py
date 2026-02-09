from langchain_core.tools import tool

# custome tools

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers and returns the result."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers and returns the result."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers and returns the result."""
    return a * b

class MathToolkit:
    """A toolkit for performing basic math operations."""
    def get_tools(self):
        return [add, subtract, multiply]
    
toolkit=MathToolkit()

tools=toolkit.get_tools()

for tool in tools:
    print("Tool name:", tool.name)
    print("Tool description:", tool.description)
    print("Tool args:", tool.args)
    print()

resultadd = add.invoke({"a": 5, "b": 10})
print("Result of addition:", resultadd)

resultsubtract = subtract.invoke({"a": 5, "b": 10})
print("Result of subtraction:", resultsubtract)

resultmultiply = multiply.invoke({"a": 5, "b": 10})
print("Result of multiplication:", resultmultiply)


     
    