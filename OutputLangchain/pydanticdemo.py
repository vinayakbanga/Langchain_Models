from pydantic import BaseModel,EmailStr
from typing import Optional


class Student(BaseModel):
    name: str = "Vinayak Banga"  
    age: Optional[int] = None
    gpa: float
    # email: Optional[EmailStr] = None


new_student ={  "gpa": 3.8}

student=Student(**new_student)

print(student)