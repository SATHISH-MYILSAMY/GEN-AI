"""
Experiment: AI-Powered Q&A Generator from PDF
===============================================
Modern LangChain v0.3+ patterns using:
  - langchain_community.document_loaders.PyPDFLoader  (replaces langchain.document_loaders)
  - langchain_text_splitters.TokenTextSplitter        (replaces langchain.text_splitter)
  - langchain_openai.ChatOpenAI                       (replaces langchain.chat_models.ChatOpenAI)
  - langchain_openai.OpenAIEmbeddings                 (replaces langchain.embeddings.openai)
  - langchain_community.vectorstores.FAISS            (replaces langchain.vectorstores)
  - langchain_core.documents.Document                 (replaces langchain.docstore.document)
  - load_summarize_chain via LCEL                     (replaces langchain.chains.summarize)
  - RetrievalQA via LCEL                              (replaces langchain.chains.RetrievalQA)

Install:
  pip install langchain langchain_community langchain_openai
  pip install faiss-cpu tiktoken pypdf python-dotenv

Data:
  Place your PDF in ./data/SDG.pdf (or update PDF_PATH below)
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)             # suppresses all deprecation warnings including asyncio ones
logging.getLogger("transformers").setLevel(logging.ERROR)                  # suppresses transformers library warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"                     # suppresses advisory warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"]            = "false"                  # suppresses parallelism warnings from tokenizers

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"                 # OpenAI API key: https://platform.openai.com

PDF_PATH    = "D:\\AI\\AI_Training_Workspace\\06_Interview_Question_Creator_Using_OLV\\data\\SDG.pdf"                                               # path to your PDF file — update if needed
OUTPUT_FILE = "answers.txt"                                                # file where generated Q&A will be saved

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

from langchain_community.document_loaders import PyPDFLoader               # ✅ replaces langchain.document_loaders.PyPDFLoader
from langchain_text_splitters import TokenTextSplitter                     # ✅ replaces langchain.text_splitter.TokenTextSplitter
from langchain_core.documents import Document                              # ✅ replaces langchain.docstore.document.Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings                  # ✅ replaces langchain.chat_models and langchain.embeddings.openai
from langchain_community.vectorstores import FAISS                         # ✅ replaces langchain.vectorstores.FAISS
from langchain_core.prompts import PromptTemplate                          # ✅ replaces langchain.prompts.PromptTemplate
from langchain_core.output_parsers import StrOutputParser                  # parses LLM output as plain string
from langchain_core.runnables import RunnablePassthrough                   # passes input unchanged through the LCEL chain

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: LOAD PDF
# ══════════════════════════════════════════════════════════════════════════
section("02: LOAD PDF")

loader = PyPDFLoader(PDF_PATH)                                             # loads the PDF from the specified path
data = loader.load()                                                        # reads and parses each page into a list of Document objects

print(f"  PDF path           : {PDF_PATH}")
print(f"  Total pages loaded : {len(data)}")                               # total number of pages extracted from the PDF
print(f"  Page 1 preview     : {data[0].page_content[:300]}")             # shows the first 300 characters of the first page


# ══════════════════════════════════════════════════════════════════════════
# 03: COMBINE ALL PAGE TEXT FOR QUESTION GENERATION
# ══════════════════════════════════════════════════════════════════════════
section("03: COMBINE ALL PAGE TEXT FOR QUESTION GENERATION")

question_gen = ""
for page in data:
    question_gen += page.page_content                                      # concatenates all page content into one large string for question generation

print(f"  Total characters combined : {len(question_gen)}")               # total number of characters across all pages
print(f"  Preview                   : {question_gen[:300]}")              # shows the first 300 characters of the combined text


# ══════════════════════════════════════════════════════════════════════════
# 04: SPLIT TEXT FOR QUESTION GENERATION (large chunks)
# ══════════════════════════════════════════════════════════════════════════
section("04: SPLIT TEXT FOR QUESTION GENERATION (large chunks)")

splitter_ques_gen = TokenTextSplitter(
    model_name="gpt-3.5-turbo",                                           # tokenizer model used to count tokens accurately
    chunk_size=10000,                                                      # large chunks — each fed to LLM for question generation
    chunk_overlap=200                                                      # overlapping tokens between chunks to preserve context
)
chunk_ques_gen = splitter_ques_gen.split_text(question_gen)                # splits the combined text into large token-based chunks

print(f"  Chunk size         : 10000 tokens")
print(f"  Chunk overlap      : 200 tokens")
print(f"  Total chunks       : {len(chunk_ques_gen)}")                    # number of large chunks created for question generation
print(f"  Chunk type         : {type(chunk_ques_gen[0])}")                # confirms chunks are plain strings


# ══════════════════════════════════════════════════════════════════════════
# 05: CONVERT CHUNKS TO DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════
section("05: CONVERT CHUNKS TO DOCUMENTS")

document_ques_gen = [Document(page_content=t) for t in chunk_ques_gen]    # wraps each text chunk in a LangChain Document object for use in chains

print(f"  Total documents    : {len(document_ques_gen)}")                 # number of Document objects created from question-gen chunks
print(f"  Document type      : {type(document_ques_gen[0])}")             # confirms objects are LangChain Document instances
print(f"  Doc[0] preview     : {document_ques_gen[0].page_content[:200]}") # shows the first 200 characters of the first document


# ══════════════════════════════════════════════════════════════════════════
# 06: SPLIT DOCUMENTS FOR ANSWER GENERATION (small chunks)
# ══════════════════════════════════════════════════════════════════════════
section("06: SPLIT DOCUMENTS FOR ANSWER GENERATION (small chunks)")

splitter_ans_gen = TokenTextSplitter(
    model_name="gpt-3.5-turbo",                                           # tokenizer model used to count tokens accurately
    chunk_size=1000,                                                       # small chunks — stored in vector DB for precise answer retrieval
    chunk_overlap=100                                                      # overlapping tokens to avoid losing context at chunk boundaries
)
document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)  # splits the large document chunks into smaller ones for RAG retrieval

print(f"  Chunk size         : 1000 tokens")
print(f"  Chunk overlap      : 100 tokens")
print(f"  Total chunks       : {len(document_answer_gen)}")               # number of smaller chunks created for answer generation


# ══════════════════════════════════════════════════════════════════════════
# 07: SETUP LLM FOR QUESTION GENERATION
# ══════════════════════════════════════════════════════════════════════════
section("07: SETUP LLM FOR QUESTION GENERATION")

llm_ques_gen_pipeline = ChatOpenAI(
    model="gpt-3.5-turbo",                                                # ✅ replaces deprecated ChatOpenAI from langchain.chat_models
    temperature=0.3                                                        # moderate temperature for creative but consistent question generation
)
print(f"  Model      : gpt-3.5-turbo")
print(f"  Temperature: 0.3")


# ══════════════════════════════════════════════════════════════════════════
# 08: DEFINE PROMPTS FOR QUESTION GENERATION
# ══════════════════════════════════════════════════════════════════════════
section("08: DEFINE PROMPTS FOR QUESTION GENERATION")

subsection("Initial Question Generation Prompt")
prompt_template = """
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the coders or programmers for their tests.
Make sure not to lose any important information.

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]                                               # 'text' will be filled with each document chunk during chain execution
)
print("  Initial question prompt defined.")

subsection("Refine Prompt (improves questions using additional context)")
refine_template = """
You are an expert at creating practice questions based on coding material and documentation.
Your goal is to help a coder or programmer prepare for a coding test.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],                          # 'existing_answer' holds previous questions; 'text' is the new chunk
    template=refine_template,
)
print("  Refine prompt defined.")


# ══════════════════════════════════════════════════════════════════════════
# 09: GENERATE QUESTIONS USING LCEL REFINE CHAIN
#     load_summarize_chain removed in langchain v1.x — replaced with LCEL
# ══════════════════════════════════════════════════════════════════════════
section("09: GENERATE QUESTIONS USING LCEL REFINE CHAIN")

print("  Running question generation (this may take a moment)...")

# Step 1: Generate initial questions from the first chunk
initial_chain = PROMPT_QUESTIONS | llm_ques_gen_pipeline | StrOutputParser()
ques = initial_chain.invoke({"text": document_ques_gen[0].page_content})   # generates initial questions from the first document chunk
print(f"  Chunk 1/{len(document_ques_gen)} processed")

# Step 2: Refine questions using each subsequent chunk
refine_chain = REFINE_PROMPT_QUESTIONS | llm_ques_gen_pipeline | StrOutputParser()

for i, doc in enumerate(document_ques_gen[1:], start=2):                  # iterates over remaining chunks to progressively refine the questions
    ques = refine_chain.invoke({
        "existing_answer": ques,                                           # current questions passed as context for refinement
        "text": doc.page_content                                           # new chunk used to refine or extend the existing questions
    })
    print(f"  Chunk {i}/{len(document_ques_gen)} processed")

print(f"\n  Generated Questions:\n")
print(ques)


# ══════════════════════════════════════════════════════════════════════════
# 10: SETUP VECTOR STORE FOR ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════════════
section("10: SETUP VECTOR STORE FOR ANSWER GENERATION")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")              # ✅ replaces OpenAIEmbeddings() with no model — now uses latest model

vector_store = FAISS.from_documents(document_answer_gen, embeddings)       # embeds all answer-gen chunks and stores them in an in-memory FAISS index

print(f"  Embedding model    : text-embedding-3-small")
print(f"  Documents embedded : {len(document_answer_gen)}")               # confirms all small chunks have been embedded and stored


# ══════════════════════════════════════════════════════════════════════════
# 11: SETUP LLM AND RAG CHAIN FOR ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════════════
section("11: SETUP LLM AND RAG CHAIN FOR ANSWER GENERATION")

llm_answer_gen = ChatOpenAI(
    model="gpt-3.5-turbo",                                                # ✅ replaces deprecated ChatOpenAI from langchain.chat_models
    temperature=0.1                                                        # low temperature for factual, consistent answers
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})              # creates a retriever that fetches top 3 most relevant chunks per query

rag_prompt = PromptTemplate.from_template(
    "Answer the question based only on the context below.\n\n"
    "Context: {context}\n\nQuestion: {question}"                           # prompt template that injects retrieved context and the question
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)                  # joins retrieved chunks into a single context string for the prompt

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} # retriever fetches relevant chunks; question passes through unchanged
    | rag_prompt                                                            # fills in the prompt with context and question
    | llm_answer_gen                                                        # sends the filled prompt to the LLM for answer generation
    | StrOutputParser()                                                     # extracts the plain text answer from the LLM response
)

print("  RAG chain built successfully.")
print(f"  LLM model          : gpt-3.5-turbo | Temperature: 0.1")
print(f"  Retriever top-k    : 3")


# ══════════════════════════════════════════════════════════════════════════
# 12: PARSE QUESTIONS AND GENERATE ANSWERS
# ══════════════════════════════════════════════════════════════════════════
section("12: PARSE QUESTIONS AND GENERATE ANSWERS")

ques_list = [q.strip() for q in ques.split("\n") if q.strip()]            # splits the generated questions string into individual questions, removing empty lines

print(f"  Total questions parsed : {len(ques_list)}")
print(f"\n  Generating answers and saving to '{OUTPUT_FILE}'...\n")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)                                                 # removes existing answers file to start fresh on each run

for i, question in enumerate(ques_list):
    print(f"  Q{i+1}: {question}")
    answer = rag_chain.invoke(question)                                    # ✅ replaces deprecated answer_generation_chain.run(question)
    print(f"  A{i+1}: {answer}")
    print(f"  {'─'*60}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:                   # appends each Q&A pair to the output file
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"{'─'*60}\n\n")

print(f"\n  All answers saved to '{OUTPUT_FILE}'")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")