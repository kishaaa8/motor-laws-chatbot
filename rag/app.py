"""
Streamlit chatbot for Motor Laws PDF + optional challan PDF.
"""

import streamlit as st

import src.config
from src.ingestion import load_pdf, load_uploaded_pdf_bytes
from src.rag_chain import create_rag_chain
from src.text_splitter import split_documents
from src.vector_store import create_chroma, get_embeddings, get_retriever


st.set_page_config(page_title="Motor Laws Chatbot", page_icon="ML", layout="wide")
st.title("Motor Laws Chatbot")
st.caption("Ask questions about the Motor Vehicles Act, or upload your challan PDF to receive answers based on both the law and your document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_signature" not in st.session_state:
    st.session_state.source_signature = None


@st.cache_resource
def cached_embeddings():
    return get_embeddings()


@st.cache_resource
def motor_laws_chunks():
    docs = load_pdf()
    return split_documents(docs, chunk_size=1000, chunk_overlap=200)


@st.cache_resource
def build_vector_db(include_challan, challan_name="", challan_bytes=b""):
    chunks = list(motor_laws_chunks())

    if include_challan and challan_bytes:
        challan_docs = load_uploaded_pdf_bytes(challan_bytes, challan_name or "challan.pdf")
        if challan_docs:
            chunks.extend(split_documents(challan_docs, chunk_size=1000, chunk_overlap=200))

    embeddings = cached_embeddings()

    return create_chroma(chunks, embeddings)


def format_chat_history(messages, limit=8):
    recent_messages = messages[-limit:]
    lines = []

    for message in recent_messages:
        role = message["role"].capitalize()
        content = message["content"]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def current_source_signature(include_challan, challan_file):
    if not include_challan or challan_file is None:
        return (False, "", b"")

    challan_bytes = challan_file.getvalue()
    challan_name = challan_file.name
    return (True, challan_name, challan_bytes)


with st.sidebar:
    st.subheader("Data Sources")
    include_challan = st.toggle("Include challan PDF", value=False)

    challan_file = None
    if include_challan:
        challan_file = st.file_uploader("Upload challan PDF", type=["pdf"])
        if challan_file is None:
            st.info("Upload a challan PDF to answer using both sources.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask your question")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    source_signature = current_source_signature(include_challan, challan_file)

    with st.spinner("Retrieving and generating answer..."):
        vector_db = build_vector_db(*source_signature)
        retriever = get_retriever(vector_db, k=4)
        rag_chain = create_rag_chain(retriever)

        result = rag_chain.invoke(
            {
                "input": user_question,
                "chat_history": format_chat_history(st.session_state.messages[:-1]),
            }
        )

    assistant_answer = result.get("answer", "No answer generated.")
    st.session_state.messages.append({"role": "assistant", "content": assistant_answer})
    st.session_state.source_signature = source_signature

    with st.chat_message("assistant"):
        st.markdown(assistant_answer)

        with st.expander("Retrieved Context"):
            for idx, doc in enumerate(result.get("context", []), start=1):
                source = doc.metadata.get("source_type", "unknown")
                page = doc.metadata.get("page", "N/A")
                filename = doc.metadata.get("uploaded_filename")
                line = f"Chunk {idx} | source={source} | page={page}"
                if filename:
                    line += f" | file={filename}"
                st.markdown(line)
                st.write(doc.page_content)
                st.divider()
