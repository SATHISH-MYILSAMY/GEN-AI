import os
from dotenv import load_dotenv                                             # loads environment variables from .env file

from langchain_community.document_loaders import PyPDFLoader               # ✅ replaces langchain.document_loaders.PyPDFLoader
from langchain_core.documents import Document                              # ✅ replaces langchain.docstore.document.Document
from langchain_text_splitters import TokenTextSplitter                     # ✅ replaces langchain.text_splitter.TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings                  # ✅ replaces langchain.chat_models and langchain.embeddings.openai
from langchain_core.prompts import PromptTemplate                          # ✅ replaces langchain.prompts.PromptTemplate
from langchain_core.output_parsers import StrOutputParser                  # parses LLM output as a plain string
from langchain_core.runnables import RunnablePassthrough                   # passes input unchanged through the LCEL chain
from langchain_community.vectorstores import FAISS                         # ✅ replaces langchain.vectorstores.FAISS

from src.prompt import *                                                   # imports prompt_template and refine_template from prompt.py


# ── Environment Setup ─────────────────────────────────────────────────────
load_dotenv()                                                              # loads OPENAI_API_KEY from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                              # reads the API key from environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY                             # sets it explicitly for OpenAI SDK


def file_processing(file_path):
    """
    Loads a PDF, extracts text, and splits it into:
    - document_ques_gen : large chunks (10k tokens) for question generation
    - document_answer_gen: small chunks (1k tokens) for answer retrieval via RAG
    """

    loader = PyPDFLoader(file_path)                                        # loads the PDF from the given path
    data = loader.load()                                                    # parses each page into a Document object

    question_gen = ""
    for page in data:
        question_gen += page.page_content                                  # concatenates all page text into one string for question generation

    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",                                       # tokenizer used to count tokens accurately
        chunk_size=10000,                                                  # large chunks — each fed to the LLM for question generation
        chunk_overlap=200                                                  # overlap to preserve context across chunk boundaries
    )
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)           # splits combined text into large token-based chunks

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen] # wraps each chunk in a LangChain Document for use in chains

    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",                                       # tokenizer used to count tokens accurately
        chunk_size=1000,                                                   # small chunks — stored in FAISS for precise answer retrieval
        chunk_overlap=100                                                  # overlap to avoid losing context at chunk boundaries
    )
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen) # splits large chunks into smaller ones for RAG

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    """
    Full pipeline:
    1. Loads and splits the PDF
    2. Generates questions using LCEL refine chain (replaces load_summarize_chain)
    3. Embeds answer chunks into FAISS vector store
    4. Builds RAG chain for answer generation (replaces RetrievalQA)
    Returns: (rag_chain, filtered_ques_list)
    """

    document_ques_gen, document_answer_gen = file_processing(file_path)   # loads PDF and prepares question/answer chunks

    # ── LLM for question generation ───────────────────────────────────────
    llm_ques_gen_pipeline = ChatOpenAI(
        temperature=0.3,                                                   # moderate temperature for creative but consistent question generation
        model="gpt-3.5-turbo"                                             # ✅ replaces deprecated ChatOpenAI from langchain.chat_models
    )

    # ── Prompts ───────────────────────────────────────────────────────────
    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template,
        input_variables=["text"]                                           # 'text' filled with each document chunk during chain execution
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],                      # 'existing_answer' holds prior questions; 'text' is the new chunk
        template=refine_template,
    )

    # ── LCEL Refine Chain (replaces load_summarize_chain + chain.run()) ───
    # Step 1: generate initial questions from the first chunk
    initial_chain = PROMPT_QUESTIONS | llm_ques_gen_pipeline | StrOutputParser()
    ques = initial_chain.invoke({"text": document_ques_gen[0].page_content}) # generates initial questions from the first document chunk

    # Step 2: refine questions using each subsequent chunk
    refine_chain = REFINE_PROMPT_QUESTIONS | llm_ques_gen_pipeline | StrOutputParser()
    for doc in document_ques_gen[1:]:                                      # iterates over remaining chunks to progressively refine questions
        ques = refine_chain.invoke({
            "existing_answer": ques,                                       # passes current questions as context for refinement
            "text": doc.page_content                                       # new chunk used to refine or extend the existing questions
        })

    # ── Vector Store for answer retrieval ─────────────────────────────────
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")          # ✅ replaces OpenAIEmbeddings() with no model — uses latest model
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)   # embeds all small chunks and stores them in an in-memory FAISS index

    # ── LLM for answer generation ─────────────────────────────────────────
    llm_answer_gen = ChatOpenAI(
        temperature=0.1,                                                   # low temperature for factual, consistent answers
        model="gpt-3.5-turbo"                                             # ✅ replaces deprecated ChatOpenAI from langchain.chat_models
    )

    # ── Filter valid questions (ending with ? or .) ───────────────────────
    ques_list = ques.split("\n")
    filtered_ques_list = [
        q for q in ques_list
        if q.strip().endswith("?") or q.strip().endswith(".")             # keeps only properly formed questions
    ]

    # ── LCEL RAG Chain (replaces RetrievalQA.from_chain_type) ────────────
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})          # fetches top 3 most relevant chunks per query

    rag_prompt = PromptTemplate.from_template(
        "Answer the question based only on the context below.\n\n"
        "Context: {context}\n\nQuestion: {question}"                       # prompt injects retrieved context and the question
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)              # joins retrieved chunks into a single context string

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} # retriever fetches chunks; question passes through unchanged
        | rag_prompt                                                        # fills the prompt with context and question
        | llm_answer_gen                                                    # sends filled prompt to LLM for answer generation
        | StrOutputParser()                                                 # extracts plain text answer from LLM response
    )

    return rag_chain, filtered_ques_list