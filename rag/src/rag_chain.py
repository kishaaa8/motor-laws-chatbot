"""
rag_chain.py

Responsible for:

1. Loading LLM
2. Creating prompts
3. Creating document chain
4. Creating retrieval chain
"""

from operator import itemgetter

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)


# =====================================================
# LOAD LLM
# =====================================================

def load_llm():

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

    return llm



# =====================================================
# ANSWER PROMPT
# =====================================================

def get_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
You are a legal assistant specializing in Indian Motor Vehicle laws.

Answer the user's question using ONLY the provided context.

Rules:
- Explain the answer clearly.
- Combine information from multiple context sections if required.
- If the context does not contain enough information, say:
  "I could not find this information in the provided documents."
- Do not invent sections, penalties, or legal provisions.
- Do not mention the context or documents in your answer.

Context:

{context}


Question:

{input}


Answer:
"""
    )

    return prompt



# =====================================================
# QUESTION REWRITE PROMPT
# =====================================================

def get_question_rewrite_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
Rewrite the user's question into a standalone question.

Use previous conversation only to resolve references like:
- it
- this
- that section
- above penalty
- previous document

If the question is already standalone, return it unchanged.

Conversation history:

{chat_history}


Current question:

{input}


Standalone question:
"""
    )

    return prompt



# =====================================================
# CHAINS
# =====================================================

def create_document_chain():

    llm = load_llm()

    prompt = get_prompt()

    return prompt | llm | StrOutputParser()



def create_question_rewriter():

    llm = load_llm()

    prompt = get_question_rewrite_prompt()

    return prompt | llm | StrOutputParser()



# =====================================================
# FORMAT DOCUMENTS
# =====================================================

def _format_docs(documents):

    formatted_docs = []

    for doc in documents:

        source = doc.metadata.get(
            "source_type",
            "unknown"
        )

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        filename = doc.metadata.get(
            "uploaded_filename",
            ""
        )


        formatted_docs.append(
            f"""
Source: {source}
Page: {page}
File: {filename}

Content:
{doc.page_content}
"""
        )


    return "\n\n".join(formatted_docs)



# =====================================================
# RAG CHAIN
# =====================================================

def create_rag_chain(retriever):

    question_rewriter = create_question_rewriter()

    document_chain = create_document_chain()


    def rewrite_question(payload):

        history = payload.get(
            "chat_history",
            ""
        )


        if not history:
            return payload["input"]


        rewritten = question_rewriter.invoke(
            {
                "chat_history": history,
                "input": payload["input"]
            }
        )


        return rewritten.strip()



    chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(
                rewrite_question
            )
        )

        .assign(
            context=itemgetter(
                "standalone_question"
            )
            |
            retriever
        )


        .assign(

            answer=

            RunnableLambda(

                lambda payload:

                {
                    "context":
                    _format_docs(
                        payload["context"]
                    ),

                    "input":
                    payload["standalone_question"]

                }

            )

            |
            document_chain

        )

    )


    return chain