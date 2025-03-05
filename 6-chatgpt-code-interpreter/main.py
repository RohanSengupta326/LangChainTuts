from typing import Any
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_csv_agent

from langchain_core.tools import Tool

load_dotenv()


def main():
    print("Start...")

    llm = ChatOllama(temperature=0, model="mistral")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """


    # another version of the reAct prompt by harrison chase
    # this just take another instruction input variable. 
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    # adding the instruction input variable to the reAct prompt. 
    prompt = base_prompt.partial(instructions=instructions)

    
    # PythonREPLTool : from langchain experimental package. 
    # its is a tool that allows LLM to execute python code
    # also didn't need to write tool description and name
    # cause this is already a inbuild langchain tool object. 
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # agent_executor.invoke(
    #     # the actual question from the user. 
    #     # we have install a qr code package that actually lets the llm generate qr codes
    #     # else it would have encountered some errors.
    #     # and it will generate the qr codes in our working directory.
    #     input={
    #         "input": """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    # uses pandas to read the csv file
    # so we need pandas and tabulate packages installed as those are used 
    # under the hood. 
    # then creates another tool dynamically that has more information 
    # about the data in the csv file. 
    csv_agent_executor = create_csv_agent(
        llm=llm,
        path="episode_info.csv",
        verbose=True,
    )

    # csv_agent.invoke(
    #     input={"input": "how many columns are there in file episode_info.csv"}
    # )
    # csv_agent.invoke(
    #     input={
    #         "input": "print the seasons by ascending order of the number of episodes they have"
    #     }
    # )

    ################################ Router Grand Agent ########################################################

    # The Grand Router Agent acts as a high-level coordinator that:
    # 1. Analyzes the user's input/question
    # 2. Determines which specialized agent (Python or CSV) is best suited to handle the task
    # 3. Routes the request to the appropriate agent and returns their response
    # 4. Python Agent: Handles code execution tasks (e.g., generating QR codes)
    # 5. CSV Agent: Handles data analysis on the episode_info.csv file

    # also we don't want the grand agent to actually do any of the task of the 
    # nested agents, it's only job is to route the request to the appropriate agent.

    # this method is used to wrap the python_agent_executor becuase if we 
    # directly pass the python_agent_executor.invoke() method to the tool,
    # its somehow not taking the input which is the prompt as a str.
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
                          # we mentioned that the grand agent should not accept code as input,
                          # because it's only job is to route the request to the appropriate agent.
                          # so it can only get input as code if the grand agent is the one executing the code. which we don't want. 
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)


    

    # now invoking this grand router agent, 
    # actually routes to the required agent based on the user input
    print(
        #this will routet to the csv agent.
        grand_agent_executor.invoke(
            {
                "input": "which season has the most episodes?",
            }
        )
    )

    print(
        # this will route to the python executor agent.
        grand_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`",
            }
        )
    )


if __name__ == "__main__":
    main()
