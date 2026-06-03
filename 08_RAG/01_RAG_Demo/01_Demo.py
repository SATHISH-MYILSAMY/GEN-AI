"""
RAG — URL Loader + ChromaDB + OpenAI
======================================
Retrieval-Augmented Generation pipeline using pure LCEL (no langchain.chains).

Python 3.14 compatibility note:
  - langchain.chains imports are AVOIDED — they use pydantic Chain base class
    which is broken on Python 3.14 due to typing changes
  - create_retrieval_chain / create_stuff_documents_chain are REPLACED with
    pure LCEL pipe operator | chains — fully compatible with Python 3.14

Components:
  - langchain_community.document_loaders.UnstructuredURLLoader  — loads URLs
  - langchain_text_splitters.RecursiveCharacterTextSplitter      — splits docs
  - langchain_chroma.Chroma                                      — vector store
  - langchain_openai.OpenAIEmbeddings                            — embeddings
  - langchain_openai.ChatOpenAI                                  — chat LLM
  - langchain_core.prompts.ChatPromptTemplate                    — prompt
  - langchain_core.runnables.RunnablePassthrough                 — LCEL chain
  - langchain_core.output_parsers.StrOutputParser                — parses output

Install:
  pip install langchain langchain_community langchain_openai langchain_chroma
  pip install langchain_text_splitters chromadb tiktoken unstructured python-dotenv
"""

import warnings
import logging
warnings.filterwarnings("ignore")                                               # suppresses all non-critical warnings
logging.getLogger("langchain").setLevel(logging.ERROR)                          # suppresses langchain internal warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"                                  # suppresses tokenizer parallelism warnings

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

from dotenv import load_dotenv                                                  # loads OPENAI_API_KEY from .env file
from langchain_community.document_loaders import UnstructuredURLLoader          # loads raw text content from a list of URLs
from langchain_text_splitters import RecursiveCharacterTextSplitter             # splits docs into overlapping chunks
from langchain_chroma import Chroma                                             # vector store — stores and retrieves embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI                       # OpenAI embeddings + chat LLM
from langchain_core.prompts import ChatPromptTemplate                           # structured prompt with system + human messages
from langchain_core.runnables import RunnablePassthrough                        # ✅ passes input unchanged through LCEL chain
from langchain_core.output_parsers import StrOutputParser                       # parses LLM output as plain string

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: LOAD ENVIRONMENT VARIABLES
# ══════════════════════════════════════════════════════════════════════════
section("02: LOAD ENVIRONMENT VARIABLES")

load_dotenv()                                                                   # reads OPENAI_API_KEY from .env file in current directory
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Add it to a .env file: OPENAI_API_KEY=sk-...")

print("  OPENAI_API_KEY loaded successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 03: LOAD DOCUMENTS FROM URLS
# ══════════════════════════════════════════════════════════════════════════
section("03: LOAD DOCUMENTS FROM URLS")

urls = [
    "https://www.victoriaonmove.com.au/local-removalists.html",
    "https://victoriaonmove.com.au/index.html",
    "https://victoriaonmove.com.au/contact.html",
]

print(f"  Loading content from {len(urls)} URLs...")
loader = UnstructuredURLLoader(urls=urls)                                       # fetches and parses raw text from each URL
data = loader.load()                                                            # returns a list of LangChain Document objects

print(f"  Loaded {len(data)} documents.")
for doc in data:
    print(f"    - {doc.metadata.get('source', 'unknown')} ({len(doc.page_content)} chars)")


# ══════════════════════════════════════════════════════════════════════════
# 04: SPLIT DOCUMENTS INTO CHUNKS
# ══════════════════════════════════════════════════════════════════════════
section("04: SPLIT DOCUMENTS INTO CHUNKS")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,                                                            # each chunk is at most 1000 characters
    chunk_overlap=200,                                                          # 200-character overlap preserves context between chunks
)
docs = text_splitter.split_documents(data)                                      # splits each document into overlapping chunks

print(f"  Total chunks created : {len(docs)}")
print(f"  Sample chunk preview : {docs[0].page_content[:200]}...")


# ══════════════════════════════════════════════════════════════════════════
# 05: CREATE EMBEDDINGS AND VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════
section("05: CREATE EMBEDDINGS AND VECTOR STORE")

print("  Embedding chunks with OpenAI (this may take a moment)...")
vectorstore = Chroma.from_documents(
    documents=docs,                                                             # chunked documents to embed and store
    embedding=OpenAIEmbeddings()                                                # converts each chunk to a vector using OpenAI embeddings
)
print(f"  Vector store created with {vectorstore._collection.count()} embeddings.")


# ══════════════════════════════════════════════════════════════════════════
# 06: CREATE RETRIEVER
# ══════════════════════════════════════════════════════════════════════════
section("06: CREATE RETRIEVER")

retriever = vectorstore.as_retriever(
    search_type="similarity",                                                   # cosine similarity search
    search_kwargs={"k": 3}                                                      # return top 3 most relevant chunks per query
)
print("  Retriever created (similarity search, k=3).")


# ══════════════════════════════════════════════════════════════════════════
# 07: TEST RETRIEVER DIRECTLY
# ══════════════════════════════════════════════════════════════════════════
section("07: TEST RETRIEVER DIRECTLY")

test_query = "What kind of services they provide?"
retrieved_docs = retriever.invoke(test_query)                                   # invoke() replaces deprecated get_relevant_documents()

print(f"  Query              : {test_query}")
print(f"  Documents retrieved: {len(retrieved_docs)}")
subsection("Top retrieved chunk")
print(f"  {retrieved_docs[0].page_content[:400]}...")


# ══════════════════════════════════════════════════════════════════════════
# 08: SET UP LLM
# ══════════════════════════════════════════════════════════════════════════
section("08: SET UP LLM")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",                                                      # chat model — compatible with ChatPromptTemplate
    temperature=0.4,                                                            # mild creativity — factual but not robotic
    max_tokens=500                                                              # limits response length to 500 tokens
)
print("  ChatOpenAI LLM initialised.")


# ══════════════════════════════════════════════════════════════════════════
# 09: BUILD RAG CHAIN (pure LCEL — no langchain.chains)
# ══════════════════════════════════════════════════════════════════════════
section("09: BUILD RAG CHAIN (pure LCEL)")

subsection("09a: Define Prompt Template")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),                                                  # system message — sets behaviour and injects retrieved context
    ("human", "{question}"),                                                    # human message — the user's question
])
print("  ChatPromptTemplate created.")

subsection("09b: Format retrieved docs helper")

def format_docs(docs):
    """Joins retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)                        # concatenates all retrieved chunks with blank lines between them

subsection("09c: Assemble LCEL RAG Chain")

# ✅ Pure LCEL chain — avoids langchain.chains entirely (Python 3.14 compatible)
# Flow: question → retriever fetches docs → format → prompt → LLM → parse
rag_chain = (
    {
        "context":  retriever | format_docs,                                    # retrieves relevant chunks and formats them as a string
        "question": RunnablePassthrough()                                       # passes the question through unchanged
    }
    | prompt                                                                    # fills {context} and {question} into the prompt template
    | llm                                                                       # sends the formatted prompt to ChatOpenAI
    | StrOutputParser()                                                         # extracts the plain text answer from the LLM response
)
print("  LCEL RAG chain assembled: retriever → format → prompt → LLM → parse")


# ══════════════════════════════════════════════════════════════════════════
# 10: RUN QUERIES
# ══════════════════════════════════════════════════════════════════════════
section("10: RUN QUERIES")

subsection("Query 1 — Services Provided")
query1 = "What kind of services they provide?"
answer1 = rag_chain.invoke(query1)                                              # ✅ invoke() with plain string — RunnablePassthrough handles it
print(f"  Question : {query1}")
print(f"  Answer   : {answer1}")

subsection("Query 2 — Contact Information")
query2 = "How can I contact them?"
answer2 = rag_chain.invoke(query2)
print(f"  Question : {query2}")
print(f"  Answer   : {answer2}")

subsection("Query 3 — Service Areas")
query3 = "What areas do they cover?"
answer3 = rag_chain.invoke(query3)
print(f"  Question : {query3}")
print(f"  Answer   : {answer3}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")