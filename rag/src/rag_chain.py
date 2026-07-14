"""
llm.py

This module is responsible for:

1. Loading the LLM
2. Creating the Prompt Template
3. Creating the Document Chain
4. Creating the Retrieval Chain
"""

from operator import itemgetter

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# =====================================================
# LOAD LLM
# =====================================================

def load_llm():
    """
    Loads the Groq LLM.

    You can change the model name later if required.
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    return llm


# =====================================================
# PROMPT TEMPLATE
# =====================================================

def get_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Use the conversation history when it helps resolve follow-up questions.

Answer ONLY from the provided context.

If the answer is not present in the context,
say "I don't know."

Conversation history:
{chat_history}

<context>
{context}
</context>

Question:
{input}
"""
    )

    return prompt


# =====================================================
# DOCUMENT CHAIN
# =====================================================

def create_document_chain():

    llm = load_llm()

    prompt = get_prompt()

    return prompt | llm | StrOutputParser()


def _format_docs(documents):
    """
    Convert retrieved documents into a single context string for the prompt.
    """

    return "\n\n".join(doc.page_content for doc in documents)


# =====================================================
# RETRIEVAL CHAIN
# =====================================================

def create_rag_chain(retriever):

    document_chain = create_document_chain()

    chain = RunnablePassthrough.assign(
        context=itemgetter("input") | retriever,
    ).assign(
        answer=RunnableLambda(
            lambda payload: {
                "context": _format_docs(payload["context"]),
                "input": payload["input"],
                "chat_history": payload.get("chat_history", ""),
            }
        )
        | document_chain,
    )

    return chain