from dotenv import load_dotenv
from langchain import chat_models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Test(BaseModel):
    """this is a pydantic schema"""



prompt = PromptTemplate(
    template="give me student information of a fictional student. \n {format_instruction}", 
    input_variables=[], 
    partial_variables={'format_instruction': Test.get_format_instructions()}
)