"""
Weaviate Demo with LangChain
==============================
Modern LangChain v0.3+ patterns using:
  - langchain_openai.OpenAIEmbeddings         (replaces langchain.embeddings.OpenAIEmbeddings)
  - langchain_openai.ChatOpenAI               (replaces langchain.llms.OpenAI)
  - langchain_weaviate.WeaviateVectorStore    (replaces langchain.vectorstores.Weaviate)
  - langchain_community.document_loaders.PyPDFDirectoryLoader
  - langchain_text_splitters.RecursiveCharacterTextSplitter
  - RAG via LCEL                              (replaces langchain.chains.RetrievalQA / load_qa_chain)
  - weaviate-client v4                        (replaces deprecated weaviate.Client v3)

Install:
  pip install langchain langchain_community langchain_openai langchain_weaviate
  pip install weaviate-client pypdf tiktoken

Data:
  Place your PDF files in ./pdfs/ before running.
  Sample PDFs used in the original notebook:
    - YOLO paper     : https://drive.google.com/file/d/1hPQlXrX8FbaYaLypxTmeVOFNitbBMlEE
    - Rachel Green CV: https://drive.google.com/file/d/1vILwiv6nS2wI3chxNabMgry3qnV67TxM

Setup:
  Create a free Weaviate cluster at https://console.weaviate.cloud/
  Copy your Cluster URL and API key into the variables below.
"""

import os
import re
import warnings
import logging
warnings.filterwarnings("ignore")                                     # suppresses all non-critical warnings
logging.getLogger("transformers").setLevel(logging.ERROR)                  # suppresses transformers library warnings
logging.getLogger("langchain_weaviate").setLevel(logging.CRITICAL)         # suppresses Weaviate batch insert error logs (metadata issues handled below)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"                     # suppresses advisory warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"]            = "false"                  # suppresses parallelism warnings from tokenizers

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"                 # OpenAI API key: https://platform.openai.com
WEAVIATE_URL     = "your_weaviate_cluster_url_here"                       # Weaviate cluster URL from Weaviate Cloud console (without https://)
WEAVIATE_API_KEY = "your_weaviate_api_key_here"                           # Weaviate API key from Weaviate Cloud console → API keys

PDF_DIR         = "./pdfs"                                                 # folder containing your PDF files
COLLECTION_NAME = "Chatbot"                                                # name of the Weaviate collection (replaces "class" in v3)

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

import weaviate                                                            # weaviate-client v4 — class-based API
from weaviate.classes.init import Auth                                     # ✅ replaces weaviate.auth.AuthApiKey (v3 style)
from langchain_community.document_loaders import PyPDFDirectoryLoader     # ✅ replaces langchain.document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter        # ✅ replaces langchain.text_splitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI                  # ✅ replaces langchain.embeddings and langchain.llms
from langchain_weaviate import WeaviateVectorStore                         # ✅ replaces langchain.vectorstores.Weaviate
from langchain_core.prompts import ChatPromptTemplate                      # modern prompt template for structuring LLM inputs
from langchain_core.output_parsers import StrOutputParser                  # parses LLM output as a plain string
from langchain_core.runnables import RunnablePassthrough                   # passes the question through unchanged in the LCEL chain

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: LOAD PDF FILES
# ══════════════════════════════════════════════════════════════════════════
section("02: LOAD PDF FILES")

os.makedirs(PDF_DIR, exist_ok=True)                                        # creates the pdfs folder if it doesn't already exist

loader = PyPDFDirectoryLoader(PDF_DIR)                                     # loads all PDF files found in the specified directory
data = loader.load()                                                        # reads and parses each PDF into a list of Document objects (one per page)

print(f"  PDF directory      : {PDF_DIR}")
print(f"  Total pages loaded : {len(data)}")                               # total number of pages extracted across all PDFs
if data:
    print(f"  Sample source      : {data[0].metadata['source']}")         # shows which file the first page came from
    print(f"  Page 1 preview     : {data[0].page_content[:300]}")         # shows the first 300 characters of the first page


# ══════════════════════════════════════════════════════════════════════════
# 03: SPLIT DOCUMENTS INTO CHUNKS
# ══════════════════════════════════════════════════════════════════════════
section("03: SPLIT DOCUMENTS INTO CHUNKS")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,                                                        # maximum number of characters per chunk
    chunk_overlap=20                                                       # overlapping characters between adjacent chunks to preserve context
)
text_chunks = text_splitter.split_documents(data)                          # splits each page into smaller overlapping chunks

print(f"  Total chunks created : {len(text_chunks)}")                     # total number of chunks after splitting all pages
print(f"\n  Chunk [1] preview    : {text_chunks[1].page_content[:300]}")  # shows the content of the second chunk
print(f"\n  Chunk [2] preview    : {text_chunks[2].page_content[:300]}")  # shows the content of the third chunk
print(f"\n  Chunk [3] preview    : {text_chunks[3].page_content[:300]}")  # shows the content of the fourth chunk


# ══════════════════════════════════════════════════════════════════════════
# 04: SETUP EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════
section("04: SETUP EMBEDDINGS")

embedding = OpenAIEmbeddings(model="text-embedding-3-small")               # ✅ latest embedding model (replaces default ada-002)

test_vector = embedding.embed_query("How are you!")                        # test call to verify the embedding model is working
print(f"  Embedding model    : text-embedding-3-small")
print(f"  Test vector length : {len(test_vector)}")                        # number of dimensions in the vector (1536 for text-embedding-3-small)


# ══════════════════════════════════════════════════════════════════════════
# 05: CONNECT TO WEAVIATE (weaviate-client v4)
# ══════════════════════════════════════════════════════════════════════════
section("05: CONNECT TO WEAVIATE (weaviate-client v4)")

import weaviate.classes as wvc                                             # needed for AdditionalConfig and Timeout settings

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,                                              # your Weaviate Cloud cluster URL
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),                      # API key authentication
    headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},           # passed to Weaviate for server-side OpenAI module calls
    additional_config=wvc.init.AdditionalConfig(
        timeout=wvc.init.Timeout(init=60)                                  # increases gRPC init timeout to 60s (default is too short on high-latency networks)
    ),
    skip_init_checks=False,                                                # set to True only if gRPC port 443 is blocked by your firewall
)

print(f"  Weaviate connected : {client.is_ready()}")                      # True if cluster is reachable and ready
print(f"  Cluster URL        : {WEAVIATE_URL}")


# ══════════════════════════════════════════════════════════════════════════
# 06: CLEAN METADATA + CREATE COLLECTION AND STORE VECTORS
# ══════════════════════════════════════════════════════════════════════════
section("06: CLEAN METADATA + CREATE COLLECTION AND STORE VECTORS")

# ── Fix: clean invalid metadata keys before inserting into Weaviate ───────
VALID_KEY = re.compile(r'^[_A-Za-z][_0-9A-Za-z]{0,230}$')                # Weaviate property names must match this GraphQL naming pattern

def clean_metadata(docs):
    """Removes metadata keys with invalid GraphQL names (e.g. ptex.fullbanner from LaTeX PDFs)."""
    for doc in docs:
        doc.metadata = {
            k: v for k, v in doc.metadata.items()
            if VALID_KEY.match(k)                                          # keeps only keys that are valid Weaviate property names
        }
    return docs

text_chunks = clean_metadata(text_chunks)                                  # strips invalid metadata keys (e.g. from YOLO LaTeX PDF) before inserting
print(f"  Metadata cleaned for {len(text_chunks)} chunks")                # confirms all chunks have been cleaned

subsection("Option A — Create new collection and store documents")

if client.collections.exists(COLLECTION_NAME):                            # checks if the collection already exists in Weaviate
    client.collections.delete(COLLECTION_NAME)                            # deletes it to avoid duplicate vectors on re-run
    print(f"  Deleted existing collection: '{COLLECTION_NAME}'")

vectorstore = WeaviateVectorStore.from_documents(                         # ✅ replaces manual schema creation + vectorstore.add_texts() from v3
    documents=text_chunks,                                                 # list of cleaned LangChain Document objects to embed and store
    embedding=embedding,                                                   # embedding model used to convert text into vectors
    client=client,                                                         # active Weaviate client connection
    index_name=COLLECTION_NAME,                                            # name of the Weaviate collection to create/use
    text_key="page_content",                                               # field name used to store the text content inside Weaviate
)
print(f"  Vectors stored in collection: '{COLLECTION_NAME}'")

subsection("Option B — Load existing collection (use if already populated)")
vectorstore = WeaviateVectorStore(                                       # connects to an already-populated Weaviate collection
    client=client,                                                       # active Weaviate client connection
    index_name=COLLECTION_NAME,                                          # must match the collection name used when vectors were stored
    embedding=embedding,                                                  # must match the embedding model used during ingestion
    text_key="page_content",                                             # must match the field name used during ingestion
)
print("  (Option B is commented out — uncomment to load an existing collection)")


# ══════════════════════════════════════════════════════════════════════════
# 07: SIMILARITY SEARCH
# ══════════════════════════════════════════════════════════════════════════
section("07: SIMILARITY SEARCH")

query = "what is yolo?"                                                    # query to search for the most relevant chunks in the vector store
docs = vectorstore.similarity_search(query, k=3)                           # ✅ replaces vectorstore.similarity_search(query, top_k=20)

print(f"  Query              : {query}")
print(f"  Documents returned : {len(docs)}")                               # number of relevant chunks returned by the similarity search
for i, doc in enumerate(docs):
    print(f"\n  Doc {i+1} source  : {doc.metadata.get('source', 'N/A')}") # shows which PDF file the chunk came from
    print(f"  Doc {i+1} preview : {doc.page_content[:200]}")               # shows the first 200 characters of each retrieved chunk


# ══════════════════════════════════════════════════════════════════════════
# 08: BUILD RAG CHAIN (LCEL — replaces load_qa_chain)
# ══════════════════════════════════════════════════════════════════════════
section("08: BUILD RAG CHAIN (LCEL — replaces load_qa_chain)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)                       # ✅ replaces OpenAI(openai_api_key=..., temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})               # creates a retriever that fetches the top 2 most relevant chunks per query

rag_prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below.\n\n"
    "Context: {context}\n\nQuestion: {question}"                           # prompt template that injects retrieved context and the user question
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)                  # joins retrieved chunks into a single context string for the prompt

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} # retriever fetches relevant chunks; question passes through unchanged
    | rag_prompt                                                            # fills in the prompt template with context and question
    | llm                                                                   # sends the filled prompt to the LLM for answer generation
    | StrOutputParser()                                                     # extracts the plain text answer from the LLM response
)

def process_response(query: str):
    """Run RAG chain and print answer + source documents."""
    answer = rag_chain.invoke(query)                                       # runs the full RAG pipeline to generate an answer
    source_docs = retriever.invoke(query)                                  # fetches source chunks again to display alongside the answer
    print(f"\n  Q: {query}")
    print(f"  A: {answer}")
    print(f"\n  Sources:")
    seen = set()                                                            # tracks already-printed sources to avoid printing duplicates
    for doc in source_docs:
        src = doc.metadata.get("source", "N/A")
        if src not in seen:
            print(f"    - {src}")
            seen.add(src)


# ══════════════════════════════════════════════════════════════════════════
# 09: Q&A
# ══════════════════════════════════════════════════════════════════════════
section("09: Q&A")

subsection("Query 1 — YOLO explanation")
process_response("What is YOLO?")

subsection("Query 2 — YOLOv7 model comparisons")
process_response("YOLOv7 outperforms which models?")

subsection("Query 3 — Rachel Green experience")
process_response("Rachel Green Experience")


# ══════════════════════════════════════════════════════════════════════════
# 10: INTERACTIVE Q&A LOOP
# ══════════════════════════════════════════════════════════════════════════
section("10: INTERACTIVE Q&A LOOP")

print("  Type your question and press Enter. Type 'exit' to quit.\n")

while True:
    user_input = input("  Input Prompt: ").strip()                         # reads user input and strips leading/trailing whitespace
    if user_input.lower() == "exit":                                       # exits the loop if the user types 'exit'
        print("  Exiting interactive Q&A.")
        break
    if not user_input:                                                     # skips empty input without breaking the loop
        continue
    process_response(user_input)                                           # runs the RAG chain and prints the answer with sources


# ══════════════════════════════════════════════════════════════════════════
# 11: CLEANUP — CLOSE WEAVIATE CONNECTION
# ══════════════════════════════════════════════════════════════════════════
section("11: CLEANUP — CLOSE WEAVIATE CONNECTION")

client.close()                                                             # ✅ always close the client when done — releases the gRPC connection
print("  Weaviate client connection closed.")
print("  To delete the collection, run: client.collections.delete('Chatbot')")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")