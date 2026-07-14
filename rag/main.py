"""
Main Application

This file connects all the modules together.
"""

import src.config

from src.ingestion import (
    load_text,
    load_pdf,
    load_web,
)

from src.text_splitter import split_documents

from src.vector_store import (
    get_embeddings,
    create_chroma,
    get_retriever,
    similarity_search,
)


def main():

    # --------------------------------------------------
    # TEXT
    # --------------------------------------------------

    print("=" * 50)
    print("TEXT FILE")
    print("=" * 50)

    text_docs = load_text()

    print(f"Documents Loaded : {len(text_docs)}")



    # --------------------------------------------------
    # WEBSITE
    # --------------------------------------------------

    print("\n")

    print("=" * 50)
    print("WEBSITE")
    print("=" * 50)

    web_docs = load_web()

    print(f"Documents Loaded : {len(web_docs)}")



    # --------------------------------------------------
    # PDF
    # --------------------------------------------------

    print("\n")

    print("=" * 50)
    print("PDF")
    print("=" * 50)

    pdf_docs = load_pdf()

    print(f"Pages Loaded : {len(pdf_docs)}")



    # --------------------------------------------------
    # SPLITTING
    # --------------------------------------------------

    print("\n")

    print("=" * 50)
    print("TEXT SPLITTING")
    print("=" * 50)

    chunks = split_documents(pdf_docs)

    print(f"Chunks Created : {len(chunks)}")



    # --------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------

    print("\n")

    print("=" * 50)
    print("EMBEDDINGS")
    print("=" * 50)

    embeddings = get_embeddings()

    print("Embedding Model Ready")



    # --------------------------------------------------
    # CHROMA
    # --------------------------------------------------

    print("\n")

    print("=" * 50)
    print("CHROMA DB")
    print("=" * 50)

    chroma_db = create_chroma(
        chunks,
        embeddings,
    )

    print("Chroma Created Successfully")



    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------
    
    from src.rag_chain import create_rag_chain

    print("\n")

    print("=" * 50)
    print("SIMILARITY SEARCH")
    print("=" * 50)

    query = "Who are the authors of Attention Is All You Need?"

    
    retriever = get_retriever(chroma_db)

    rag_chain = create_rag_chain(retriever)

    response = rag_chain.invoke(
        {
            "input": "Who are the authors of Attention Is All You Need?"
        }
    )

    print(response["answer"])


if __name__ == "__main__":
    main()