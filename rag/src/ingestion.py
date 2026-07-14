"""
ingestion.py

Minimal document loaders for local text, local PDF, web URL,
and uploaded PDF files.
"""

import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader


RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PDF_PATH = RAG_DIR / "motor_laws.pdf"
DEFAULT_TEXT_PATH = RAG_DIR / "guide.txt"


def load_pdf(pdf_path=None):
    """
    Load the motor laws PDF (or a provided PDF path).
    """

    target = Path(pdf_path) if pdf_path else DEFAULT_PDF_PATH

    if not target.exists():
        raise FileNotFoundError(f"PDF file not found: {target}")

    documents = PyPDFLoader(str(target)).load()

    for doc in documents:
        doc.metadata["source_type"] = "motor_laws"

    return documents


def load_text(text_path=None):
    """
    Load a local text file.
    """

    target = Path(text_path) if text_path else DEFAULT_TEXT_PATH

    if not target.exists():
        raise FileNotFoundError(f"Text file not found: {target}")

    return TextLoader(str(target), encoding="utf-8").load()


def load_web(url="https://docs.smith.langchain.com/"):
    """
    Load data from a web page URL.
    """

    return WebBaseLoader(url).load()


def load_uploaded_pdf(uploaded_file):
    """
    Load a PDF uploaded from Streamlit's file uploader.
    """

    if uploaded_file is None:
        return []

    suffix = Path(uploaded_file.name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        documents = PyPDFLoader(tmp_path).load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    for doc in documents:
        doc.metadata["source_type"] = "challan"
        doc.metadata["uploaded_filename"] = uploaded_file.name

    return documents


def load_uploaded_pdf_bytes(file_bytes, filename="uploaded.pdf"):
    """
    Load an uploaded PDF from raw bytes.
    """

    if not file_bytes:
        return []

    suffix = Path(filename).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        documents = PyPDFLoader(tmp_path).load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    for doc in documents:
        doc.metadata["source_type"] = "challan"
        doc.metadata["uploaded_filename"] = filename

    return documents
