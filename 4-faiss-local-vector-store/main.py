import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

if __name__ == "__main__":
    print("hi this is a tut on FAISS local vector database.")
    pdf_path = r"C:\Users\shawn\Documents\DevFiles\personal\LangChain(course)\4-faiss-local-vector-store\react.pdf"

    # pyPDFLoader loads and reads the pdf and put them into Langchain Document type objects
    # and create chunks of those.
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # we are splitting the doc again into chunks cause
    # we don't wanna hit the token limit of the llm
    # but still create meaningful chunks.
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # embed
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model='mistral', temperature=0, )

    # FAISS: converts docs into small vectors which can be stored in ram
    # locally .
    # good for using vector db locally.
    # supported with langchain.
    # convert the chunkified docs into vectors. and store in faiss ( locally )
    # and returns an object which is the vector Store.
    vectorstore = FAISS.from_documents(docs, embeddings)
    # normally FAISS stores docs into ram, so temporary
    # so if you call this method, it will store in local storage.
    vectorstore.save_local("faiss_index_react")

    # loads the docs from the local vector Store.
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    # RETRIEVAL:

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
