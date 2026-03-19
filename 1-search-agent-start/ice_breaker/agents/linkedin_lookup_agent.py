from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_agent
from tools.tools import get_profile_url_tavily

from langchain_ollama import ChatOllama


# -> str : hints that this method will return str type
# doesn't enforce the method to return that though.
def lookup(name: str) -> str:
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        # other params...
    )

    template = """given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
                              Your answer should contain only 1 URL strictly."""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    # list of tools for the agent to use
    tools_for_agent = [
        # from the langchain tool package
        # tools provided to the llm to actually perform something
        # on the internet or something.
        Tool(
            name="Search Google for linkedin profile page",
            # which method to run when the tool is used.
            func=get_profile_url_tavily,
            # desc: to specify for the agent when to run the tool
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    agent = create_agent(
        model=llm,
        tools=tools_for_agent,
        system_prompt=(
            "You find the correct LinkedIn profile page for a person. "
            "Use the search tool when needed and return only one URL."
        ),
    )


    """ 
    Agent Scratchpad Management:

    The agent_scratchpad variable is automatically managed by the AgentExecutor.
    It keeps track of the intermediate steps (thoughts, actions, observations) during the agent's reasoning process.
    When you use verbose=True in AgentExecutor, you can see this scratchpad being built in real-time.


    Chat History:

    If you're using the agent in a conversation, the chat_history parameter is populated automatically when you pass in previous exchanges.
    In your example, since it's a single question, this would typically be empty or contain previous turns if in a session. 
    
    """


    # the dict key name is 'input' cause its the input variable in the react prompt. 
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template.format(name_of_person=name),
                }
            ]
        }
    )

    # print(result)

    # get the output from the response
    final_message = result["messages"][-1]
    content = getattr(final_message, "content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    linked_profile_url = str(content).strip()

    # print(linked_profile_url)

    return linked_profile_url


if __name__ == "__main__":
    print(lookup(name="Rohan Sengupta Mantis Pro Gaming"))


""" 
This Agent implementation was done using reAct prompt and AgentExecutor. 
but it doesn't require you to manually call tools like in a traditional ReAct setup. Instead, the AgentExecutor handles tool execution automatically

How This ReAct Implementation Works
Agent Thinks: "I need to search for a LinkedIn profile."

Agent Writes a Thought:
Thought: I need to search Google for a LinkedIn profile.
Action: Search Google for linkedin profile page
Action Input: "Rohan Sengupta Mantis Pro Gaming"

LangChain Parses the Output → Extracts "Search Google for linkedin profile page" as the tool.
AgentExecutor Calls the Tool (get_profile_url_tavily).
Observation is Returned to the LLM → The loop repeats until the agent reaches a final answer.

"""


""" 
TOOL BINDING / FUNCTION CALLING: (implementation in langGraph course project)

Tool Binding and Agent Executor both allow an LLM to use external tools, but they work in fundamentally different ways.

Tool Binding (also called Function Calling) is a newer and more structured approach where the LLM directly generates a function call in a structured format, typically JSON. When a user asks a question, the LLM decides if it needs to call a tool and returns a structured response specifying the tool name and the parameters required. LangChain automatically executes the tool and feeds the result back to the LLM. There is no need for manual parsing or handling tool execution separately. This method is more reliable and efficient because the LLM doesn't generate unstructured text that needs to be interpreted—it simply outputs a function call, making the process seamless and reducing errors.


Agent Executor is still useful for cases where multi-step reasoning is important before deciding which tool to use

"""
