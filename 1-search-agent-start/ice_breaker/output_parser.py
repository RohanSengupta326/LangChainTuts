from typing import List, Dict, Any

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


"""
Pydantic
Pydantic is a data validation library in python that enforces type hints at runtime.
mentioning a data types of the dict. else generally python doesn't have a specific data
type style which will cause errors. 
How it works:

Define models with type annotations
Pydantic validates data when instances are created
It converts data to the right types when possible
It raises validation errors at runtime when data doesn't match
"""

# extends BaseModel class
class Summary(BaseModel):
    # these 'Field's are from pydantic and used when converting to pydantic Object.
    # meaning the normal string data will be converted to
    # a dict like object with summary key with string summary
    # and facts key : which consists of List of string .
    # means, it will simply parse the normal string data into this pydantic object.
    # similar to a json.
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}


summary_parser = PydanticOutputParser(pydantic_object=Summary)
