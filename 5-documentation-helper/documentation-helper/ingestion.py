from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OllamaEmbeddings(model="nomic-embed-text")


def ingest_docs():
    # read the docs and load into to langchain Document object.
    loader = ReadTheDocsLoader(
        r"C:\Users\shawn\Documents\DevFiles\personal\LangChain(course)\5-documentation-helper\documentation-helper\langchain-docs\api.python.langchain.com\en\latest"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    # same as previous tut projects.
    # split characters as before. but recursively in all files.
    # chunking ( Formula of cal chunk size) :
    # input and output sum of both is the total token count
    # so now, if there are 4 context windows lets assume
    # and 2k token limit
    # then each window will have 500 tokens approx.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # just add url for the actual langchain docs by iterating.
    # it would take a while to load all the vectors to pinecone.
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
