# INGESTION:
# load the data
# split the content
# embed the split content
# store the vectors in pinecone vector db

# RETRIEVAL:
# the user question is converted into vectors
# then a prompt is created with the vectors
# and given to llm
# llm can search the vectorDB for nearest vectors
# and generates response.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# LANGCHAIN DOCUMENT LOADERS:
# the langchain document_loaders are just implementations on how to handle
# different formats of text data, like from: text, whatsapp, notion , GDrive etc.
# and format it so that its easily readable by the llm. thats it.
# so there are multiple document loaders available in langchain.

# LANGCHAIN TEXT SPLITTERS:
# based on llm token limit we need to split the text into chunks.

# CHARACTER TEXT SPLITTER:
# chunk size and chunk_overlap properties are sent to the CharacterTextSplitter
# so that the text isn't split in a way which ruins its original meaning.

# OPEN AI EMBEDDINGS:
# converts text into vectors. along with meta data containing the actual text.
# openai ada 2 is the cheapest ones.

# PINECONE VECTOR STORAGE:
# texts converted to vectors we need to store in pinecone vector db.


if __name__ == '__main__':
    print("Ingesting...")
    # Langchain has multiple document loaders like this one
    # for slack, google drive, whatsapp etc
    # some are inbuilt some are community created.
    # can found usage on langchain docs => document loader.
    loader = TextLoader(r"C:\Users\shawn\Documents\DevFiles\personal\LangChain(course)\3-rag-gist\mediumblog1.txt",
                        encoding="utf-8")
    document = loader.load()

    print("splitting...")
    # split texts into chunks .
    # chunk_size = more chunk size = more money = confused llm
    # but also should be big enough to understand what the chunk means
    # to humans and llms both. else it won't generate proper response.
    # so 1000 is a standard token amount
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # embed the text into vectors. using openAI API
    # can use other embedding from ollama and others.
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Use Ollama's free embedding model instead of OpenAI
    # ollama runs locally so no api key is needed
    # you would have to pull ollama's embedding model: "nomic-embed-text"
    # on the cmd line locally though.
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("ingesting...")
    # langchain takes all the text and embeds them using the embeder by iterating.
    # and stores them in the pineconeVector db .
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finish")
