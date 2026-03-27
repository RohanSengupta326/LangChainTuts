from dotenv import load_dotenv
from typing import TypedDict, List, Literal, Annotated, Optional
from langchain import chat_models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt1 = """ Please write a detailed report on the given topic: {topic} """
prompt_template1 = PromptTemplate(template=prompt1, input_variables=['topic'])

prompt2 = "Pleae write a 1 line summary of the given text. /n {text}"
prompt_template2 = PromptTemplate(template=prompt2, input_variables=['text'])

prompt = prompt_template1.invoke({'topic': 'youtube as a career'})

result1 = model.invoke(prompt)

prompt = prompt_template2.invoke({'text': result1.content})

result2 = model.invoke(prompt)

print(result2.content)



