"""
RAG Streamlit App
==================
Python 3.14 compatible — uses pure LCEL instead of langchain.chains
which is broken on Python 3.14 due to pydantic typing changes.
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI               # ✅ ChatOpenAI replaces OpenAI (works with ChatPromptTemplate)
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter     # ✅ replaces deprecated langchain.text_splitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough                # ✅ replaces create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser               # ✅ replaces create_stuff_documents_chain

load_dotenv()                                                           # loads OPENAI_API_KEY from .env file


# ── Page config ───────────────────────────────────────────────────────────
st.title("RAG App Demo")


# ── Load and index documents (cached so it only runs once) ────────────────
@st.cache_resource
def build_rag_chain():
    urls = [
        "https://www.victoriaonmove.com.au/local-removalists.html",
        "https://victoriaonmove.com.au/index.html",
        "https://victoriaonmove.com.au/contact.html",
    ]

    # Load URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(data)

    # Embed and store in Chroma
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings()
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # LLM
    llm = ChatOpenAI(                                                   # ✅ ChatOpenAI replaces OpenAI
        model="gpt-3.5-turbo",
        temperature=0.4,
        max_tokens=500
    )

    # Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # ✅ Pure LCEL RAG chain — avoids langchain.chains (Python 3.14 compatible)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context":  retriever | format_docs,                        # retrieves + formats relevant chunks
            "question": RunnablePassthrough()                           # passes question through unchanged
        }
        | prompt                                                        # fills context + question into prompt
        | llm                                                           # sends to ChatOpenAI
        | StrOutputParser()                                             # extracts plain text answer
    )

    return rag_chain


# ── Build chain (cached) ──────────────────────────────────────────────────
with st.spinner("Loading documents and building index..."):
    rag_chain = build_rag_chain()


# ── Chat input ────────────────────────────────────────────────────────────
query = st.chat_input("Ask me anything:")

if query:
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(query)                                # ✅ plain string invoke — RunnablePassthrough handles it
    st.write(answer)
    
    
# RUN QUERIES
# ======================================================================

#   ── Query 1 — Services Provided ──
#   Question : What kind of services they provide?
#   Answer   : They provide comprehensive local moving services for residents and businesses in Melbourne, including apartment moving, villa moving, household moving, office moving, furniture moving, and optional packing and unpacking services. They also offer customised moving plans tailored to specific requirements, schedules, and budgets. Additionally, every move includes transit and public liability insurance to protect goods from loading through to delivery.

#   ── Query 2 — Contact Information ──
#   Question : How can I contact them?
#   Answer   : You can contact Victoria On Move by phone at 0404 922 328 or via email at victoriaonmove07@gmail.com. They are available 24/7 for inquiries and to provide personalized moving quotes. Their location is in Wollert, Victoria, Australia.

#   ── Query 3 — Service Areas ──
#   Question : What areas do they cover?
#   Answer   : Victoria On Move offers local moving services for Melbourne residents and businesses, covering the Melbourne area and potentially other parts of Australia. They serve clients with professionalism and care, focusing on seamless moving and packing solutions since 2024.