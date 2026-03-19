from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_agent
from tools.tools import get_profile_url_tavily


def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
    )
    template = """
       given the name {name_of_person} I want you to find a link to their Twitter profile page, and extract from it their username
       In Your Final answer only the person's username"""
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Twitter Page URL",
        )
    ]

    agent = create_agent(
        model=llm,
        tools=tools_for_agent,
        system_prompt=(
            "You find a person's Twitter or X profile. "
            "Use the search tool when needed and return only the username."
        ),
    )

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

    final_message = result["messages"][-1]
    content = getattr(final_message, "content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    twitter_username = str(content).strip()
    return twitter_username


if __name__ == "__main__":
    print(lookup(name="Elon Musk"))
