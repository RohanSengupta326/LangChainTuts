from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Student(BaseModel):
    name: str
    roll: int = 1
    id: int = '001'
    aadhar: Optional[str] = 0000
    age: int = Field(lt=19, gt=5, description="this is the age of the new student",)
    email: Optional[EmailStr]



# student = Student(name='rohan sengupta', id=1234, age=18)

student_info_dict = {'name': 'rohan sengupta', 'id':1234, 'age':18, 'email': 'rohan@gmail.com'}

student = Student(**student_info_dict)

print(student)

