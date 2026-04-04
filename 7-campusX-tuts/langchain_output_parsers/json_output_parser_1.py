from dotenv import load_dotenv
from langchain import chat_models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


outputParser = JsonOutputParser()

prompt = PromptTemplate(
    template="give me student information of a fictional student. \n {format_instruction}", 
    input_variables=[], 
    partial_variables={'format_instruction': outputParser.get_format_instructions()}
)


chain = prompt | model |  outputParser

result = chain.invoke({})

print(result)
