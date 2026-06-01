"""
TinyLlama 1.1B Chat + ChromaDB Multi-Document Retriever with LangChain
========================================================================
Modern LangChain v0.3+ patterns using:
  - langchain_community.vectorstores.Chroma
  - langchain_huggingface.HuggingFaceEmbeddings
  - langchain_huggingface.HuggingFacePipeline
  - langchain_community.document_loaders
  - langchain_text_splitters.RecursiveCharacterTextSplitter
  - LCEL RetrievalQA chain (replaces deprecated RetrievalQA.from_chain_type)

NOTE: Switched from falcon-7b to TinyLlama-1.1B-Chat — runs on CPU with no
      disk offload issues. Sample stock market articles downloaded automatically.

Install:
  pip install langchain langchain_community langchain_huggingface
  pip install transformers accelerate torch
  pip install sentence_transformers chromadb tiktoken pypdf sentencepiece
"""

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"
os.environ["CLEAN_UP_TOKENIZATION_SPACES"]      = "false"

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

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain_chroma import Chroma 
# from langchain_community.vectorstores import Chroma                            # ✅ replaces langchain.vectorstores.Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader   # ✅ replaces langchain.document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter            # ✅ replaces langchain.text_splitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings  # ✅ replaces langchain.llms / langchain.embeddings
from langchain_core.prompts import PromptTemplate                              # ✅ modern prompt template
from langchain_core.runnables import RunnablePassthrough                       # ✅ LCEL passthrough for chain building
from langchain_core.output_parsers import StrOutputParser                      # ✅ parses LLM output as plain string

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: DOWNLOAD SAMPLE DATA
# ══════════════════════════════════════════════════════════════════════════
section("02: DOWNLOAD SAMPLE DATA")

import urllib.request
import zipfile

data_url = "https://github.com/Shafi2016/Youtube/raw/main/stock_market_june_2023.zip"
zip_path = "stock_market_june_2023.zip"
data_dir = "./stock_market_june_2023/"

if not os.path.exists(data_dir):
    print(f"  Downloading sample data from GitHub...")
    urllib.request.urlretrieve(data_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")
    os.remove(zip_path)
    print(f"  Data downloaded and extracted to: {data_dir}")
else:
    print(f"  Data directory already exists, skipping download: {data_dir}")


# ══════════════════════════════════════════════════════════════════════════
# 03: LOAD AND SPLIT DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════
section("03: LOAD AND SPLIT DOCUMENTS")

subsection("03a: Load Text Files from Directory")
loader = DirectoryLoader(
    data_dir,
    glob="./*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}           # ✅ handles non-UTF-8 characters in source files
)
documents = loader.load()
print(f"  Loaded {len(documents)} documents from {data_dir}")

subsection("03b: Split Documents into Chunks")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,                                                            # each chunk max 500 characters
    chunk_overlap=100                                                          # 100-character overlap between chunks
)
texts = text_splitter.split_documents(documents)
print(f"  Split into {len(texts)} text chunks")
print(f"  Sample chunk: {texts[2].page_content[:200]}...")


# ══════════════════════════════════════════════════════════════════════════
# 04: LOAD TINYLLAMA MODEL AND PIPELINE
# ══════════════════════════════════════════════════════════════════════════
section("04: LOAD TINYLLAMA MODEL AND PIPELINE")

subsection("04a: Load Tokenizer")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"                               # ✅ instruction-tuned, runs on CPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token                    = tokenizer.eos_token
tokenizer.clean_up_tokenization_spaces = False
print(f"  Tokenizer loaded for model: {model_id}")

subsection("04b: Load Model on CPU")
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU detected : {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    print(f"  Loading on   : CPU (forced — insufficient VRAM)")
else:
    print(f"  No GPU detected — loading on CPU")

model_obj = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,                                                       # ✅ float32 — stable on CPU
    device_map="cpu",                                                          # ✅ force CPU — avoids disk offload
    offload_folder="offload_cache",                                            # ✅ safe fallback
)

# ✅ Replace entire generation_config — eliminates stale max_length conflict
model_obj.generation_config = GenerationConfig(
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    repetition_penalty=1.1,
    max_new_tokens=300,                                                        # ✅ set on model — no pipeline conflict
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
model_obj.config.max_length = None                                             # ✅ clears stale max_length=20

print("  Model loaded successfully on CPU.")

subsection("04c: Create Text Generation Pipeline")
pipeline = transformers.pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    return_full_text=False,                                                    # ✅ returns only new tokens
    device=-1,                                                                 # ✅ CPU
)
print("  Pipeline created successfully.")

subsection("04d: Wrap Pipeline as LangChain LLM")
llm = HuggingFacePipeline(pipeline=pipeline)
print("  HuggingFacePipeline LLM created.")


# ══════════════════════════════════════════════════════════════════════════
# 05: CREATE EMBEDDINGS AND VECTOR DATABASE
# ══════════════════════════════════════════════════════════════════════════
section("05: CREATE EMBEDDINGS AND VECTOR DATABASE")

subsection("05a: Load Embedding Model")
embedding_model = "intfloat/e5-large-v2"                                       # strong retrieval embeddings ~1.3GB
hf = HuggingFaceEmbeddings(model_name=embedding_model)
print(f"  Embedding model loaded: {embedding_model}")

subsection("05b: Create and Persist Chroma Vector Database")
persist_directory = "db"

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=hf,
    persist_directory=persist_directory                                        # ✅ auto-persists in Chroma >= 0.4.0
)
print(f"  Vector database created with {vectordb._collection.count()} embeddings")
print(f"  Persisted to: {persist_directory}/")

subsection("05c: Reload Vector Database from Disk")
vectordb = None
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=hf
)
print(f"  Vector database reloaded from disk: {persist_directory}/")


# ══════════════════════════════════════════════════════════════════════════
# 06: BUILD RETRIEVAL QA CHAIN (LCEL — replaces RetrievalQA.from_chain_type)
# ══════════════════════════════════════════════════════════════════════════
section("06: BUILD RETRIEVAL QA CHAIN (LCEL)")

retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}                                                     # top 3 most similar chunks per query
)

# ✅ TinyLlama ChatML format with RAG context injected
rag_template = """<|system|>
You are a helpful assistant. Use only the provided context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>"""

rag_prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context", "question"]
)

def format_docs(docs):
    """Joins retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_sources(docs):
    """Extracts unique source file paths from retrieved documents."""
    return list({doc.metadata.get("source", "unknown") for doc in docs})

# ✅ LCEL chain — replaces deprecated RetrievalQA.from_chain_type
#    flow: question → retriever → format_docs → prompt → llm → parse
rag_chain = (
    {
        "context" : retriever | format_docs,                                   # retrieves chunks and joins them
        "question": RunnablePassthrough()                                       # passes question through unchanged
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("  LCEL RAG chain built successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 07: HELPER — PROCESS AND PRINT RESPONSE WITH SOURCES
# ══════════════════════════════════════════════════════════════════════════
section("07: HELPER FUNCTION")

def ask(query):
    """Runs the RAG chain and prints the answer with source documents."""
    print(f"  Query   : {query}")
    answer  = rag_chain.invoke(query)
    sources = get_sources(retriever.invoke(query))
    print(f"\n  Answer  : {answer.strip()}")
    print(f"\n  Sources :")
    for s in sources:
        print(f"    - {s}")


# ══════════════════════════════════════════════════════════════════════════
# 08: RUN QUERIES
# ══════════════════════════════════════════════════════════════════════════
section("08: RUN QUERIES")

subsection("Query 1 — Companies with Potential Stock Growth")
ask("Could you please enumerate the companies highlighted for their potential stock growth")

subsection("Query 2 — Microsoft's Investment in OpenAI")
ask("How much has Microsoft invested in OpenAI?")

subsection("Query 3 — Shopify Layoffs")
ask("What were the reasons behind Shopify's decision to lay off a portion of its workforce")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")