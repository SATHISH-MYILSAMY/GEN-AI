"""
Pinecone Demo with LangChain
==============================
Modern LangChain v0.3+ patterns using:
  - langchain_openai.OpenAIEmbeddings       (replaces langchain.embeddings.OpenAIEmbeddings)
  - langchain_openai.ChatOpenAI             (replaces langchain.llms.OpenAI)
  - langchain_pinecone.PineconeVectorStore  (replaces langchain.vectorstores.Pinecone)
  - langchain_community.document_loaders.PyPDFDirectoryLoader
  - langchain_text_splitters.RecursiveCharacterTextSplitter
  - RAG via LCEL                            (replaces langchain.chains.RetrievalQA)
  - pinecone v3                             (replaces deprecated pinecone.init() / pinecone-client v2)

Install:
  pip install langchain langchain_community langchain_openai langchain_pinecone
  pip install pinecone pypdf tiktoken

Data:
  Place your PDF files in ./pdfs/ before running.
  Sample PDFs used in the original notebook:
    - YOLOv7 paper   : https://drive.google.com/file/d/1hPQlXrX8FbaYaLypxTmeVOFNitbBMlEE
    - Rachel Green CV: https://drive.google.com/file/d/1vILwiv6nS2wI3chxNabMgry3qnV67TxM
"""

import os
import sys
import warnings
import logging
warnings.filterwarnings("ignore")                                          # suppresses all non-critical warnings
logging.getLogger("transformers").setLevel(logging.ERROR)                  # suppresses transformers library warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"                     # suppresses advisory warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"]            = "false"                  # suppresses parallelism warnings from tokenizers

os.environ["OPENAI_API_KEY"]   = "your_openai_api_key_here"               # OpenAI API key: https://platform.openai.com
os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key_here"             # Pinecone API key: Pinecone console → API keys (no environment needed in v3)

PDF_DIR    = "./pdfs"                                                      # folder containing your PDF files
INDEX_NAME = "test"                                                        # use your existing index name shown in the Pinecone console

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

from pinecone import Pinecone, ServerlessSpec                              # pinecone v3 class-based API — no pinecone.init() needed
from langchain_community.document_loaders import PyPDFDirectoryLoader     # ✅ replaces langchain.document_loaders.PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter        # ✅ replaces langchain.text_splitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI                  # ✅ replaces langchain.embeddings and langchain.llms
from langchain_pinecone import PineconeVectorStore                         # ✅ pinecone v3 compatible vector store
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
# 05: INITIALIZE PINECONE (pinecone v3)
# ══════════════════════════════════════════════════════════════════════════
section("05: INITIALIZE PINECONE (pinecone v3)")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])                      # creates Pinecone client — no environment param needed in v3

existing_indexes = [idx.name for idx in pc.list_indexes()]                 # fetches all existing index names in your Pinecone project

if INDEX_NAME not in existing_indexes:                                     # only creates the index if it doesn't already exist
    pc.create_index(
        name=INDEX_NAME,                                                   # name of the new index to create
        dimension=1536,                                                    # must match embedding model output dimension (1536 for text-embedding-3-small)
        metric="cosine",                                                   # cosine similarity is recommended for text embeddings
        spec=ServerlessSpec(cloud="aws", region="us-east-1")               # matches your Pinecone console: Region aws us-east-1, Type Dense
    )
    print(f"  Created new index   : '{INDEX_NAME}'")
else:
    print(f"  Using existing index: '{INDEX_NAME}'")                       # your 'test' index already exists so this line will print

print(f"  All indexes         : {existing_indexes}")                       # lists all indexes currently in your Pinecone project


# ══════════════════════════════════════════════════════════════════════════
# 06: EMBED CHUNKS AND STORE IN PINECONE
# ══════════════════════════════════════════════════════════════════════════
section("06: EMBED CHUNKS AND STORE IN PINECONE")

subsection("Option A — Create new vector store from documents")
index = pc.Index(INDEX_NAME)                                               # connects to the Pinecone index by name

docsearch = PineconeVectorStore(
    index=index,                                                           # the Pinecone index object to store and search vectors
    embedding=embedding,                                                   # embedding model used to convert text chunks into vectors
)
docsearch.add_documents(text_chunks)                                       # embeds all text chunks and upserts them into the Pinecone index
print(f"  Vectors stored in index: '{INDEX_NAME}'")

subsection("Option B — Load existing index (use if already populated)")
# docsearch = PineconeVectorStore(
#     index=pc.Index(INDEX_NAME),                                          # connects to an already-populated Pinecone index
#     embedding=embedding                                                  # must match the embedding model used when the index was created
# )
print("  (Option B is commented out — uncomment to load an existing index)")


# ══════════════════════════════════════════════════════════════════════════
# 07: SIMILARITY SEARCH
# ══════════════════════════════════════════════════════════════════════════
section("07: SIMILARITY SEARCH")

query = "YOLOv7 outperforms which models"                                  # query to search for the most relevant chunks in the vector store
docs = docsearch.similarity_search(query, k=3)                             # retrieves the top 3 most semantically similar chunks

print(f"  Query              : {query}")
print(f"  Documents returned : {len(docs)}")                               # number of relevant chunks returned by the similarity search
for i, doc in enumerate(docs):
    print(f"\n  Doc {i+1} source  : {doc.metadata.get('source', 'N/A')}") # shows which PDF file the chunk came from
    print(f"  Doc {i+1} preview : {doc.page_content[:200]}")               # shows the first 200 characters of each retrieved chunk


# ══════════════════════════════════════════════════════════════════════════
# 08: BUILD RAG CHAIN (LCEL — replaces RetrievalQA)
# ══════════════════════════════════════════════════════════════════════════
section("08: BUILD RAG CHAIN (LCEL — replaces RetrievalQA)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)                       # ✅ replaces OpenAI() — temperature=0 gives deterministic answers

retriever = docsearch.as_retriever(search_kwargs={"k": 2})                 # creates a retriever that fetches the top 2 most relevant chunks per query

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

subsection("Query 1 — YOLOv7 model comparisons")
process_response("YOLOv7 outperforms which models")

subsection("Query 2 — Rachel Green experience")
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


section("ALL SECTIONS COMPLETED SUCCESSFULLY")
