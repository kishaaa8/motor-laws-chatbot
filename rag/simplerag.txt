"""
=========================================================
Document Loaders + Text Splitting + Vector Database Demo
=========================================================

This script demonstrates:

1. Loading data from a text file
2. Loading data from a website
3. Loading data from a PDF
4. Splitting documents into smaller chunks
5. Creating embeddings using OpenAI
6. Storing embeddings inside Chroma DB
7. Performing similarity search
8. Creating a FAISS vector database


"""

# =====================================================
# IMPORTS
# =====================================================

import os
import bs4

from dotenv import load_dotenv

# Document Loaders
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
)

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Vector Databases
from langchain_community.vectorstores import Chroma, FAISS


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

# Loads variables from the .env file
load_dotenv()

# Store the OpenAI API key in environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():

    # =================================================
    # 1. LOAD A TEXT FILE
    # =================================================

    print("\n========== TEXT FILE ==========\n")

    text_loader = TextLoader("speech.txt")

    text_documents = text_loader.load()

    print(f"Loaded {len(text_documents)} document(s).")
    print(text_documents)


    # =================================================
    # 2. LOAD CONTENT FROM A WEBSITE
    # =================================================

    print("\n========== WEBSITE ==========\n")

    web_loader = WebBaseLoader(

        web_paths=(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
        ),

        # BeautifulSoup only extracts these HTML classes.
        # This avoids loading unnecessary page content.

        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-title", "post-content", "post-header")
            )
        ),
    )

    web_documents = web_loader.load()

    print(f"Loaded {len(web_documents)} web document(s).")


    # =================================================
    # 3. LOAD A PDF
    # =================================================

    print("\n========== PDF ==========\n")

    pdf_loader = PyPDFLoader("attention.pdf")

    pdf_documents = pdf_loader.load()

    print(f"PDF contains {len(pdf_documents)} pages.")


    # =================================================
    # 4. SPLIT DOCUMENTS INTO SMALLER CHUNKS
    # =================================================

    """
    Large Language Models have context limits.

    Instead of giving an entire PDF,
    we divide it into smaller overlapping chunks.

    chunk_size = maximum size of one chunk

    chunk_overlap = keeps some common text between
    neighbouring chunks so that context isn't lost.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = splitter.split_documents(pdf_documents)

    print(f"Total chunks created : {len(documents)}")

    print("\nFirst Chunk:\n")
    print(documents[0].page_content)


    # =================================================
    # 5. CREATE EMBEDDINGS
    # =================================================

    """
    Embeddings convert text into vectors.

    Similar meaning -> Similar vectors.

    These vectors are stored inside a Vector Database.
    """

    embeddings = OpenAIEmbeddings()


    # =================================================
    # 6. CREATE CHROMA VECTOR DATABASE
    # =================================================

    print("\n========== CHROMA ==========\n")

    chroma_db = Chroma.from_documents(
        documents,
        embeddings,
    )

    print("Chroma database created successfully.")


    # =================================================
    # 7. SIMILARITY SEARCH
    # =================================================

    query = "Who are the authors of Attention Is All You Need?"

    results = chroma_db.similarity_search(query)

    print("\nMost Relevant Result:\n")

    print(results[0].page_content)


    # =================================================
    # 8. CREATE FAISS VECTOR DATABASE
    # =================================================

    print("\n========== FAISS ==========\n")

    """
    FAISS is another vector database.

    Here we only use the first 15 chunks
    to make the example faster.
    """

    faiss_db = FAISS.from_documents(
        documents[:15],
        embeddings,
    )

    print("FAISS database created successfully.")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()