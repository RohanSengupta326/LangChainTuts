from dotenv import load_dotenv
from langchain import chat_models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    """Person Schema representing details of a fictional person"""
    name: str = Field(description="The name of the man")
    age: int = Field(lt=50, gt=18, description="Age of the person")
    city: str = Field(description="city where he is from")


pydanticOutputParser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="give me information of a fictional person from {country}. \n {format_instruction}", 
    input_variables=['country'], 
    partial_variables={'format_instruction': pydanticOutputParser.get_format_instructions()}
)

chain = prompt | model | pydanticOutputParser


result = chain.invoke({'country': "Zimbabwe"})

print(result)