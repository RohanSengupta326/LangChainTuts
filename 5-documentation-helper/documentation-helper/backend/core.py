import os

from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama

from typing import List, Dict, Any

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = os.environ["INDEX_NAME"]


# similar retrieval logic as shown in previous tut projects.
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOllama(model="mistral", verbose=True, temperature=0)

    # this prompt actually rephrases the question with the previous
    # question and generated response. that is with the memory.
    # and creates a new standalone question.
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # takes the llm as it will do some llm call
    # takes the context from the vector storage and plugs it in,
    # in the prompt.
    # this is the 2nd step :
    # putting the context vector fetched from the vector DB
    # and send the prompt to the llm.
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # adds the standalone question to the retrieval prompt.
    # and sends to llm as memory.
    # and the retrieval search happens according to new standalone question.
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    # converts user input to vector and retrieve similar vector from the db.
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # adding the chat_history
    # this serves as the memory of the llm.
    # so that it can remember the conversation that it had.
    # we can only send the chat_history in the format of list of dictionaries.
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    # just created a new dictionary with different key names.
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }

    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["result"])
