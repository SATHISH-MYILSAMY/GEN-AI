"""
ChromaDB Demo with LangChain
==============================
Modern LangChain v0.3+ patterns using:
  - langchain_openai.OpenAIEmbeddings   (replaces langchain.embeddings.OpenAIEmbeddings)
  - langchain_openai.ChatOpenAI         (replaces langchain.llms.OpenAI)
  - langchain_community.vectorstores.Chroma
  - langchain_community.document_loaders.DirectoryLoader / TextLoader
  - langchain_text_splitters.RecursiveCharacterTextSplitter
  - RetrievalQA via LCEL                (replaces langchain.chains.RetrievalQA)

Install:
  pip install langchain langchain_community langchain_openai chromadb tiktoken

Data:
  Download articles zip: https://www.dropbox.com/s/vs6ocyvpzzncvwh/new_articles.zip
  Extract to: ./new_articles/
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR) # suppresses warnings from transformers library about model loading
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" # suppresses the new advisory warnings from transformers library about model loading
os.environ["TOKENIZERS_PARALLELISM"]            = "false" # suppresses parallelism warnings from tokenizers library

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"   # https://platform.openai.com

# ── Update this to your local articles folder ─────────────────────────────
ARTICLES_DIR = "./new_articles"   # ← update path if needed
DB_DIR       = "./db" # directory where Chroma DB will be saved

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

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI                  # ✅ replaces langchain.embeddings / langchain.llms
from langchain_community.document_loaders import DirectoryLoader, TextLoader  # ✅ replaces langchain.document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter        # ✅ replaces langchain.text_splitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
section("02: LOAD DATA")

subsection("Loading .txt files from directory")
loader = DirectoryLoader(
    ARTICLES_DIR,
    glob="*.txt", # built-in tool used to search for file pathnames
    loader_cls=TextLoader, # built-in loader for text files
    loader_kwargs={"encoding": "utf-8"}, # ensures text files are read with UTF-8 encoding (handles most text files correctly)
    silent_errors=True # skips any problematic files gracefully
)
documents = loader.load() # loads all .txt files from the specified directory, creating a list of Document objects
print(f"  Total documents loaded : {len(documents)}") # prints the total number of documents successfully loaded from the directory
print(f"  Sample source          : {documents[0].metadata['source']}") # prints the source (file path) of the first loaded document to verify that loading worked correctly
print(f"  Sample content preview : {documents[0].page_content[:200]}") # prints the first 200 characters of the content of the first loaded document to give a preview of the data that was loaded


# ══════════════════════════════════════════════════════════════════════════
# 03: SPLIT DOCUMENTS INTO CHUNKS
# ══════════════════════════════════════════════════════════════════════════
section("03: SPLIT DOCUMENTS INTO CHUNKS")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # splits text into chunks of 1000 characters with 200 characters overlap (to preserve context across chunks)
chunks = text_splitter.split_documents(documents) # splits each document into multiple chunks, each with its own metadata (including source)

print(f"  Total chunks created : {len(chunks)}") # prints the total number of chunks created 
print(f"\n  Chunk [1] preview    :")
print(f"  {chunks[1].page_content[:300]}") # prints the first 300 characters of the content of the second chunk
print(f"\n  Chunk [2] preview    :")
print(f"  {chunks[2].page_content[:300]}") # prints the first 300 characters of the content of the third chunk


# ══════════════════════════════════════════════════════════════════════════
# 04: CREATE CHROMA VECTOR DB
# ══════════════════════════════════════════════════════════════════════════
section("04: CREATE CHROMA VECTOR DB")

subsection("Embedding model + Chroma DB setup")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")   # ✅ latest embedding model

vectordb = Chroma.from_documents(
    documents=chunks, # list of text chunks to be embedded and stored in the vector DB
    embedding=embedding, # embedding function to convert text chunks into vectors
    persist_directory=DB_DIR # directory where Chroma DB will be saved on disk (creates folder if it doesn't exist
)
print(f"  Chroma DB created at  : {DB_DIR}") # prints the directory where the Chroma DB has been created
print(f"  Total vectors stored  : {vectordb._collection.count()}") # prints the total number of vectors stored in the Chroma DB

# Persist to disk
print("  DB persisted to disk.") # since Chroma uses an in-memory collection, the data is not actually saved to disk until we call persist() or delete the object.


# ══════════════════════════════════════════════════════════════════════════
# 05: RELOAD DB FROM DISK
# ══════════════════════════════════════════════════════════════════════════
section("05: RELOAD DB FROM DISK")

vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding
)
print(f"  Chroma DB reloaded from: {DB_DIR}")
print(f"  Total vectors in DB    : {vectordb._collection.count()}")


# ══════════════════════════════════════════════════════════════════════════
# 06: MAKE A RETRIEVER
# ══════════════════════════════════════════════════════════════════════════
section("06: MAKE A RETRIEVER")

retriever = vectordb.as_retriever(search_kwargs={"k": 2}) # creates a retriever that can be used to query the vector DB; search_kwargs={"k": 2} specifies that the top 2 most relevant chunks should be retrieved for each query
print(f"  Search type   : {retriever.search_type}") # prints the type of search algorithm used by the retriever
print(f"  Search kwargs : {retriever.search_kwargs}") # prints the search parameters used by the retriever

subsection("Test retrieval — 'How much money did Microsoft raise?'")
docs = retriever.invoke("How much money did Microsoft raise?") # retrieves the top 2 most relevant chunks from the vector DB based on the query
print(f"  Documents retrieved: {len(docs)}") 
for i, doc in enumerate(docs):
    print(f"\n  Doc {i+1} source  : {doc.metadata['source']}") # prints the source (file path) of each retrieved document to verify that the retriever is returning relevant results
    print(f"  Doc {i+1} preview : {doc.page_content[:200]}") # prints the first 200 characters of the content of each retrieved document to give a preview of the results returned by the retriever


# ══════════════════════════════════════════════════════════════════════════
# 07: BUILD RAG CHAIN (LCEL)
# ══════════════════════════════════════════════════════════════════════════
section("07: BUILD RAG CHAIN (LCEL — replaces RetrievalQA)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # ✅ temperature=0 for deterministic output (good for RAG)

rag_prompt = ChatPromptTemplate.from_template( # creates a prompt template for the RAG chain that includes the retrieved context and the user's question
    "Answer the question based only on the context below. "
    "If you don't know the answer, say 'I don't know'.\n\n"
    "Context: {context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} # defines the inputs to the RAG chain: 'context' is generated by passing the query through the retriever and formatting the retrieved documents, while 'question' is passed through unchanged
    | rag_prompt # formats the prompt with the retrieved context and the question
    | llm # generates the answer based on the formatted prompt
    | StrOutputParser() # extracts the answer as a string from the LLM's output
)

def process_response(query: str):
    """Run RAG chain and print answer + source documents."""
    answer = rag_chain.invoke(query) # runs the RAG chain with the user's query to get the answer
    source_docs = retriever.invoke(query) # retrieves the source documents again to display alongside the answer 
    print(f"\n  Q: {query}")
    print(f"  A: {answer}")
    print(f"\n  Sources:")
    seen = set() # to track and avoid duplicate sources in case the same document is retrieved multiple times
    for doc in source_docs:
        src = doc.metadata["source"]
        if src not in seen:
            print(f"    - {src}")
            seen.add(src)


# ══════════════════════════════════════════════════════════════════════════
# 08: QUERY THE RAG CHAIN
# ══════════════════════════════════════════════════════════════════════════
section("08: QUERY THE RAG CHAIN")

subsection("Query 1 — Microsoft funding")
process_response("How much money did Microsoft raise?")

subsection("Query 2 — Pando news")
process_response("What is the news about Pando?")

subsection("Query 3 — AI and open source")
process_response("What are the key concerns around open source AI models?")


# ══════════════════════════════════════════════════════════════════════════
# 09: CLEANUP — DELETE CHROMA DB
# ══════════════════════════════════════════════════════════════════════════
section("09: CLEANUP — DELETE CHROMA DB")

import gc
import subprocess
import sys

# Step 1: Delete the collection and release the object
vectordb.delete_collection()
del vectordb
gc.collect()

print("  ChromaDB collection deleted.")

# Step 2: Delete the folder using a separate process (bypasses Windows in-process lock)
if os.path.exists(DB_DIR):
    abs_db_dir = os.path.abspath(DB_DIR)
    result = subprocess.run(
        ["powershell", "-Command", f"Remove-Item -Recurse -Force '{abs_db_dir}'"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"  DB directory '{DB_DIR}' removed.")
    else:
        print(f"  Warning: Could not remove DB directory.")
        print(f"  Run manually: Remove-Item -Recurse -Force '{abs_db_dir}'")

print("\n  Cleanup complete.")

print("\n  To reload DB from a saved zip:")
print("  import shutil; shutil.unpack_archive('db.zip', '.')")
print("  vectordb = Chroma(persist_directory='./db', embedding_function=embedding)")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")