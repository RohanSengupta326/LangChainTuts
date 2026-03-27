from dotenv import load_dotenv
from typing import List, Literal, Annotated, Optional
from langchain import chat_models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    summary: str
    topics: List[str]
    rating: float

    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(description='analyse the sentiment of the review is positive, negative or neutral')
    pros: Optional[list[str]] = Field(default=None, description='If the review has some pros explicity mentioned as pros then consider those and return')
    cons: Optional[list[str]] = Field(default=None, description='If the review has some cons explicity mentioned as cons then consider those and return')
    reviewer: Optional[str] = Field(default=None, description='If the name of reviewer is present in the review then return that')


structured_output_model = model.with_structured_output(schema=Review, method='json_schema')


result = structured_output_model.invoke(""" Review: • Design and Build: Premium and stylish design. Build quality feels solid. ⭐️⭐️⭐️⭐️⭐️
• Comfort and Fit: Very lightweight and comfortable for long use. ⭐️⭐️⭐️⭐️⭐️
• Sound Quality: Rich sound with strong bass and clear vocals. Well balanced audio. ⭐️⭐️⭐️⭐️⭐️
• Active Noise Cancellation: Works well and reduces most background noise, but not complete silence. Still very effective for daily use. ⭐️⭐️⭐️⭐️
• Transparency Mode: Useful and clear when needing to hear surroundings. ⭐️⭐️⭐️⭐️⭐️
• Battery Life: Excellent backup, easily lasts a full day. ⭐️⭐️⭐️⭐️⭐️
• Fast Charging: Very fast and convenient. ⭐️⭐️⭐️⭐️⭐️
• Connectivity: Instant pairing and stable connection with no drops. ⭐️⭐️⭐️⭐️⭐️
• Touch Controls: Smooth and accurate response. ⭐️⭐️⭐️⭐️⭐️
• Call Quality: Clear voice with good noise handling. ⭐️⭐️⭐️⭐️⭐️
• Value for Money: Great features at this price range. ⭐️⭐️⭐️⭐️⭐️ 
                                        
Review by : Rohan Sengupta
                                        """)


print(result)
print(type(result))


