"""
RAG — PDF Loader + ChromaDB + OpenAI
======================================
Retrieval-Augmented Generation pipeline using pure LCEL (no langchain.chains).

Components:
  - langchain_community.document_loaders.PyPDFLoader       — loads PDF pages
  - langchain_text_splitters.RecursiveCharacterTextSplitter — splits docs
  - langchain_chroma.Chroma                                 — vector store
  - langchain_openai.OpenAIEmbeddings                       — text-embedding-3-small
  - langchain_openai.ChatOpenAI                             — gpt-4o-mini
  - langchain_core.runnables.RunnablePassthrough            — LCEL chain
  - langchain_core.output_parsers.StrOutputParser           — parses output

Requires:
  - OpenAI API key in a .env file: OPENAI_API_KEY=sk-...
  - Get your key at: https://platform.openai.com/api-keys
  - A PDF file at the path set in pdf_path below

Install:
  pip install langchain langchain_community langchain_openai langchain_chroma
  pip install langchain_text_splitters chromadb pypdf tiktoken python-dotenv
"""

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════
# 01: IMPORTS
# ══════════════════════════════════════════════════════════════════════════
section("01: IMPORTS")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader                    # loads PDF files page by page
from langchain_text_splitters import RecursiveCharacterTextSplitter             # splits docs into overlapping chunks
from langchain_chroma import Chroma                                             # vector store
from langchain_openai import OpenAIEmbeddings                                   # OpenAI embedding model
from langchain_openai import ChatOpenAI                                         # OpenAI chat LLM
from langchain_core.prompts import ChatPromptTemplate                           # structured prompt template
from langchain_core.runnables import RunnablePassthrough                        # passes question unchanged through chain
from langchain_core.output_parsers import StrOutputParser                       # parses LLM output as plain string

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: LOAD ENVIRONMENT VARIABLES
# ══════════════════════════════════════════════════════════════════════════
section("02: LOAD ENVIRONMENT VARIABLES")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it to a .env file: OPENAI_API_KEY=sk-...\n"
        "Get your key at: https://platform.openai.com/api-keys"
    )

print("  OPENAI_API_KEY loaded successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 03: LOAD PDF
# ══════════════════════════════════════════════════════════════════════════
section("03: LOAD PDF")

pdf_path = "D:\\AI\\AI_Training_Workspace\\08_RAG\\03_RAG_OpenAI\\01_RAG_ChromaDB_OpenAI\\pdf\\my_paper.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"PDF not found: {pdf_path}\n"
        f"Place your PDF file at: {os.path.abspath(pdf_path)}"
    )

loader = PyPDFLoader(pdf_path)
data = loader.load()                                                            # list of Document objects (one per page)

print(f"  PDF loaded    : {pdf_path}")
print(f"  Total pages   : {len(data)}")
print(f"  Sample content: {data[0].page_content[:200]}...")


# ══════════════════════════════════════════════════════════════════════════
# 04: SPLIT DOCUMENTS INTO CHUNKS
# ══════════════════════════════════════════════════════════════════════════
section("04: SPLIT DOCUMENTS INTO CHUNKS")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,                                                            # max 1000 characters per chunk
    chunk_overlap=200,                                                          # 200-char overlap preserves context
)
docs = text_splitter.split_documents(data)

print(f"  Total chunks created : {len(docs)}")
print(f"  Sample chunk preview : {docs[0].page_content[:200]}...")


# ══════════════════════════════════════════════════════════════════════════
# 05: INITIALISE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════
section("05: INITIALISE EMBEDDINGS")

# text-embedding-3-small — fast, cheap, 1536-dim, strong performance
# Alternative: text-embedding-3-large (3072-dim, more accurate, costs more)
EMBEDDING_MODEL = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=api_key,
)
print(f"  Embedding model : {EMBEDDING_MODEL}")


# ══════════════════════════════════════════════════════════════════════════
# 06: TEST EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════
section("06: TEST EMBEDDINGS")

test_vector = embeddings.embed_query("hello, world!")
print(f"  Vector dimensions : {len(test_vector)}")
print(f"  Sample values     : {test_vector[:5]}")


# ══════════════════════════════════════════════════════════════════════════
# 07: CREATE VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════
section("07: CREATE VECTOR STORE")

print("  Embedding chunks with OpenAI (this may take a moment)...")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
)

print(f"  Vector store created with {vectorstore._collection.count()} embeddings.")


# ══════════════════════════════════════════════════════════════════════════
# 08: CREATE RETRIEVER
# ══════════════════════════════════════════════════════════════════════════
section("08: CREATE RETRIEVER")

retriever = vectorstore.as_retriever(
    search_type="similarity",                                                   # cosine similarity search
    search_kwargs={"k": 10}                                                     # return top 10 most relevant chunks
)
print("  Retriever created (similarity search, k=10).")


# ══════════════════════════════════════════════════════════════════════════
# 09: TEST RETRIEVER DIRECTLY
# ══════════════════════════════════════════════════════════════════════════
section("09: TEST RETRIEVER DIRECTLY")

test_query = "What is new in this paper?"
retrieved_docs = retriever.invoke(test_query)

print(f"  Query              : {test_query}")
print(f"  Documents retrieved: {len(retrieved_docs)}")
subsection("Sample retrieved chunk (index 0)")
print(f"  {retrieved_docs[0].page_content[:400]}...")


# ══════════════════════════════════════════════════════════════════════════
# 10: SET UP OPENAI LLM
# ══════════════════════════════════════════════════════════════════════════
section("10: SET UP OPENAI LLM")

# gpt-4o-mini — fast, cheap, excellent for RAG Q&A
# Alternatives: "gpt-4o" (smarter, costlier), "gpt-3.5-turbo" (legacy)
LLM_MODEL = "gpt-4o-mini"

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.3,                                                            # low temperature for factual answers
    max_tokens=500,                                                             # limit response length
    openai_api_key=api_key,
)
print(f"  ChatOpenAI LLM initialised ({LLM_MODEL}).")


# ══════════════════════════════════════════════════════════════════════════
# 11: BUILD RAG CHAIN (pure LCEL — no langchain.chains)
# ══════════════════════════════════════════════════════════════════════════
section("11: BUILD RAG CHAIN (pure LCEL)")

subsection("11a: Define Prompt Template")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),                                                  # injects retrieved context
    ("human", "{question}"),                                                    # user's question
])
print("  ChatPromptTemplate created.")

subsection("11b: Format retrieved docs helper")

def format_docs(docs):
    """Joins retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

subsection("11c: Assemble LCEL RAG Chain")

# Flow: question → retriever → format → prompt → LLM → parse
rag_chain = (
    {
        "context":  retriever | format_docs,                                    # fetch + format chunks
        "question": RunnablePassthrough()                                       # pass question unchanged
    }
    | prompt                                                                    # fill {context} and {question}
    | llm                                                                       # call OpenAI
    | StrOutputParser()                                                         # extract plain text
)
print(f"  LCEL RAG chain assembled: retriever → format → prompt → {LLM_MODEL} → parse")


# ══════════════════════════════════════════════════════════════════════════
# 12: RUN QUERIES
# ══════════════════════════════════════════════════════════════════════════
section("12: RUN QUERIES")

subsection("Query 1 — Key Contributions")
query1 = "What is new in this paper?"
answer1 = rag_chain.invoke(query1)
print(f"  Question : {query1}")
print(f"  Answer   : {answer1}")

subsection("Query 2 — Methodology")
query2 = "What methodology or approach does this paper use?"
answer2 = rag_chain.invoke(query2)
print(f"  Question : {query2}")
print(f"  Answer   : {answer2}")

subsection("Query 3 — Results and Findings")
query3 = "What are the main results or findings of this paper?"
answer3 = rag_chain.invoke(query3)
print(f"  Question : {query3}")
print(f"  Answer   : {answer3}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")