"""
RAG — PDF Loader + ChromaDB + OpenAI — Streamlit App
======================================================
Streamlit chatbot interface for RAG pipeline using OpenAI.

Components:
  - langchain_community.document_loaders.PyPDFLoader       — loads PDF pages
  - langchain_text_splitters.RecursiveCharacterTextSplitter — splits docs
  - langchain_chroma.Chroma                                 — vector store
  - langchain_openai.OpenAIEmbeddings                       — text-embedding-3-small
  - langchain_openai.ChatOpenAI                             — gpt-4o-mini
  - langchain_core (pure LCEL)                              — no langchain.chains

Requires:
  - OPENAI_API_KEY in a .env file
  - PDF file path set in PDF_PATH below

Run:
  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter             # ✅ not langchain.text_splitter (deprecated)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Config ────────────────────────────────────────────────────────────────
PDF_PATH       = r"D:\\AI\\AI_Training_Workspace\\08_RAG\\03_RAG_OpenAI\\02_RAG_Q&A_Streamlit\\pdf\\my_paper.pdf"                                                 # ← change to your PDF path
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"
RETRIEVER_K     = 10

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat — OpenAI",
    page_icon="📄",
    layout="centered",
)

st.title("📄 RAG Application")
st.caption(f"PDF: `{PDF_PATH}` · Embeddings: `{EMBEDDING_MODEL}` · LLM: `{LLM_MODEL}`")

# ── Load API key ──────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to a .env file: OPENAI_API_KEY=sk-...")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# Build RAG pipeline — cached so it only runs once per session
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading PDF and building vector store...")
def build_rag_chain():
    # 1. Load PDF
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF not found: {PDF_PATH}")
        st.stop()

    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(data)

    # 3. Embed and store in ChromaDB
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    # 4. Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    # 5. LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=500,
        openai_api_key=api_key,
    )

    # 6. Prompt
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. Pure LCEL chain — no langchain.chains (Python 3.14 compatible)
    rag_chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, len(docs)


rag_chain, num_chunks = build_rag_chain()
st.success(f"✅ Vector store ready — {num_chunks} chunks indexed.")

# ══════════════════════════════════════════════════════════════════════════
# Chat interface
# ══════════════════════════════════════════════════════════════════════════

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept new input
if query := st.chat_input("Ask me anything about the PDF..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and stream answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(query)
        st.markdown(answer)

    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    

# ── Query 1 — Key Contributions ──
#   Question : What is new in this paper?
#   Answer   : The paper presents a systematic investigation into rainfall measurement problems, highlighting systematic errors in traditional precipitation estimation methods. It introduces multiple combined regression methods for rainfall prediction and compares the performance of various machine learning models. Additionally, it outlines future work focused on real-life applications and the use of neural network-based approaches to enhance prediction accuracy.

#   ── Query 2 — Methodology ──
#   Question : What methodology or approach does this paper use?
#   Answer   : The paper employs a methodology that includes four significant steps: data collection, data pre-processing, training models using ten supervised regressors, and performance evaluation. It utilizes a dataset from the Kaggle platform, which is split into training and validation parts, and applies various regression techniques such as linear, tree-based, and ensemble models. The performance of the models is compared using statistical assessment measurements to determine the best performer.

#   ── Query 3 — Results and Findings ──
#   Question : What are the main results or findings of this paper?
#   Answer   : The main findings of the paper indicate that the Random Forest regressor outperformed other models in predicting rainfall, achieving an R² score of approximately 0.87 and a low error rate compared to models like Gradient Boosting. The study utilized various regression techniques and concluded that all models showed acceptable performance, but Random Forest was the most efficient for this use case. Additionally, the research highlighted the importance of data collection and pre-processing in enhancing model performance.