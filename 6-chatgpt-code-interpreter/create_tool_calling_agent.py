# Create_tool_calling_agent implementation.

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearchResults(), multiply]
    llm = ChatOllama(model="mistral", temperature=0)
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)


    # Function calling is when an LLM can:
    # 1. Understand what functions are available
    # 2. Choose the right function to use
    # 3. but it can't call it automatically. 
    # 4. we have to manually call the function with the proper parameters.
    
    # create_tool_calling_agent improves on basic function calling by:
    # 1. Handling multiple tools/functions automatically
    # 2. Managing the conversation flow
    # 3. Providing better error handling
    # 4. Giving cleaner, more structured outputs
    
    # ReAct (Reasoning + Acting) agents are different because they:
    # 1. Give you more control over the prompt template
    # 2. Let you customize how tools are used
    # 3. Show their reasoning process (thought, action, observation)
    # 4. Allow more complex multi-step reasoning
    
    # create_tool_calling_agent does similar things but:
    # 1. Handles the complexity internally
    # 2. Uses a simpler, more streamlined approach
    # 3. Works well for straightforward tool usage
    # 4. Requires less configuration
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    res = agent_executor.invoke(
        {
            "input": "what is the weather in dubai right now? compare it with San Fransisco, output should in in celcius",
        }
    )

    print(res)
