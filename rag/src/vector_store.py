"""
vector_store.py

This module is responsible for:

1. Creating embeddings
2. Creating Chroma DB
3. Creating Retriever
4. Running similarity search
"""

from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Creates the embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_chroma(documents, embeddings):
    """
    Creates a Chroma Vector Database.
    """

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

def get_retriever(db, k=4):
    """
    Returns an improved retriever from the vector database.
    """

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )
def similarity_search(db, query):
    """
    Performs similarity search on the vector database.
    """

    return db.similarity_search(query)