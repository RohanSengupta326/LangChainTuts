import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model='mistral', temperature=0,)

    query = "what is Pinecone in machine learning?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # this is langchain hub prompt, which tells the llm to answer
    # the user's question, based only
    # on context windows information that's available.
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # these two methods below stuff the documents in the prompt.
    # meaning this handles the conversion of query to vectors
    # and formatting it to send to vector db to search similar vectors.
    # then retrieves similar vectors by searching with the vector
    # combine_docs_chain.
    # and then sends it to the llm context.
    # then llm generates response.
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result)
