from dotenv import load_dotenv
import os

from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI  # Still using LangChain's OpenAI wrapper

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser

# use the linkedin scrape that we have implemented ourselves.
from third_parties.linkedin import scrape_linkedin_profile

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets

from output_parser import summary_parser, Summary

from langchain_core.output_parsers import JsonOutputParser

from typing import Tuple

# before python 3.8 , couldn't have mentioned type like this tuple[type1, type2]
# had to import
# from typing import Tuple
# then Tuple[type1, type2]
def ice_break_with(name: str) -> Tuple[Summary, str]:
    print("Hello LangChain with OpenRouter")

    # print(os.environ['OPENAI_API_KEY'])

    # with os module we can access the env variables
    # print(os.environ['COOL_API_KEY'])

    # Prompt template
    summary_template = """
        given the information about a person from linkedin {information},
        and their latest twitter posts {twitter_posts} I want you to create:
        
        1. A short summary
        2. two interesting facts about them 

        Use both information from twitter and Linkedin
        
        
        Format the response as a JSON object matching this schema: {format_instructions}
        """

    # langchain's prompt template to create prompt with
    # input variable & base prompt.
    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    # ChatOpenAI is a langchain wrapper which wraps the process of using
    # api keys of openAI
    # similarly it has other wrappers to wrap the usage of other orgs api keys
    # anthropic, deepseek, google etc.

    # Using OpenRouter instead of OpenAI
    # hence had to update the OPENAI_API_BASE
    # in the .env file to redirect to openrouter and not openAI
    # as langchain doesn't support openRouter directly. so
    # had to bypass the base url.
    llm = ChatOpenAI(
        temperature=0, model="cognitivecomputations/dolphin3.0-r1-mistral-24b:free"
    )

    # USAGE OF OLLAMA LLM model. with langchain.
    # downloaded latest version of ollama locally llama3.2 first.
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        # other params...
    )

    # langchain is supported with the llm providers like
    # ollama :  let us use llm models locally
    # openAI : let us use llm models on cloud with api.

    # pull mistral from cmdline first
    # ollama pull mistral
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        # other params...
    )

    # Create a LangChain pipeline (| operator passes input from prompt to model)
    # chain = summary_prompt_template | llm

    # to parse the output from LLM to desired output type.
    # in this case the output is coming like: content=""
    # so we extract it and show
    # chain = summary_prompt_template | llm | StrOutputParser()

    # similarly pass the output to summary_parser to parse the output into pydantic object.
    # so that it's usable in frontEnd and stuff.
    # also the summary_parser=PydanticOutputParser takes a json object,
    # so the llm has to return the json structure else it will throw error.
    # so you have mention it to the llm in the PromptTemplate so that it can do it .
    chain = summary_prompt_template | llm | summary_parser

    # used the lambda method, cause the response of the llm is coming in a dict, and content has the actual response in json format that I wanted, so have to extract it and send to summary_parser.
    # chain = summary_prompt_template | llm | (lambda x: x.content) | summary_parser

    # Example LinkedIn data (commented out actual scraping)
    linkedin_data = """Rohan Sengupta
    (He/Him)
    SDE-1 @ Mantis Pro Gaming
    """

    # agentic search to find linkedin profile url
    linkedin_username = linkedin_lookup_agent(name=name)
    # send the url to scrape linkedin to get the data.
    # linkedin_data = scrape_linkedin_profile(
    #     linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True
    # )
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username, mock=True
    )

    # if u want to scrape twitter too.
    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username, mock=True)

    # actually invoke the api call to get the information from the linkedin data.
    res: Summary = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})

    # returns a tuple.
    return res, linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # INTEGRATE WITH LANGSMITH USING LANGSMITH API KEYS TO VISUALLY SEE EACH STEP IN THE WHOLE LLM AND AGENT EXECUTION PROCESS. FOR DEBUGGING PURPOSES.

    print("Ice Breaker Enter")
    print(ice_break_with(name="Rohan Sengupta Mantis Pro Gaming")[0])
