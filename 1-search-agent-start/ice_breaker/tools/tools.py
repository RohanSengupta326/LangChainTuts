from langchain_community.tools.tavily_search import TavilySearchResults


# have to add tavily api key in the .env file.
def get_profile_url_tavily(name: str):
    """Searches for Linkedin or Twitter Profile Page."""
    # tavily searches the internet with a prompt and returns response in json.
    # its integrated with langchain.
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res[0]["url"]
