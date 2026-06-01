"""
LangChain Overview
==================
Modern LangChain v0.3+ patterns using:
  - ChatOpenAI / ChatHuggingFace (replaces legacy OpenAI LLM wrapper)
  - LCEL pipe operator |  (replaces LLMChain, SequentialChain)
  - langgraph create_react_agent  (replaces initialize_agent / AgentType)
  - RunnableWithMessageHistory    (replaces ConversationBufferMemory / ConversationChain)

References:
  GitHub : https://github.com/langchain-ai/langchain
  Docs   : https://python.langchain.com/docs/introduction/
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["OPENAI_API_KEY"]                    = "your_open-ai_api_key_here"
os.environ["HUGGINGFACEHUB_API_TOKEN"]          = "your_huggingface_api_token_here"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════
# 03: LARGE LANGUAGE MODELS (Chat Models)
# ══════════════════════════════════════════════════════════════════════════
section("03: LARGE LANGUAGE MODELS (Chat Models)")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

subsection("OpenAI Example 1")
response = llm.invoke("What would be a good company name for a company that makes colorful socks?")
print(response.content)

subsection("OpenAI Example 2")
response = llm.invoke("I want to open a restaurant for Indian food. Suggest a fancy name for this.")
print(response.content)

subsection("HuggingFace Example 1")
from huggingface_hub import InferenceClient

hf_client = InferenceClient(
    model="meta-llama/Llama-3.2-1B-Instruct",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
response = hf_client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a translator. Output only the translated text, nothing else."},
        {"role": "user",   "content": "Translate English to German: How old are you?"}
    ],
    max_tokens=64,
)
print(response.choices[0].message.content)

subsection("HuggingFace Example 2")
response = hf_client.chat_completion(
    messages=[{"role": "user", "content": "I want to open a restaurant for Indian food. Suggest a fancy name."}],
    max_tokens=256,
)
print(response.choices[0].message.content)

subsection("HuggingFace via LCEL")
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def hf_invoke(prompt_value) -> str:
    text = prompt_value.text if hasattr(prompt_value, "text") else str(prompt_value)
    resp = hf_client.chat_completion(
        messages=[{"role": "user", "content": text}],
        max_tokens=512,
    )
    return resp.choices[0].message.content

hf_llm = RunnableLambda(hf_invoke)

hf_chain = (
    PromptTemplate.from_template("I want to open a restaurant for {cuisine} food. Suggest a fancy name.")
    | hf_llm
    | StrOutputParser()
)
print(hf_chain.invoke({"cuisine": "Mexican"}))


# ══════════════════════════════════════════════════════════════════════════
# 04: PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════
section("04: PROMPT TEMPLATES")

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

subsection("PromptTemplate with input_variables")
prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)
print(prompt_template_name.format(cuisine="Indian"))

subsection("PromptTemplate.from_template")
prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
print(prompt.format(product="colorful socks"))

subsection("ChatPromptTemplate (recommended for chat models)")
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful business consultant."),
    ("human",  "I want to open a restaurant for {cuisine} food. Suggest a fancy name."),
])
for msg in chat_prompt.format_messages(cuisine="Mexican"):
    print(msg)


# ══════════════════════════════════════════════════════════════════════════
# 05: CHAINS (LCEL)
# ══════════════════════════════════════════════════════════════════════════
section("05: CHAINS (LCEL – LangChain Expression Language)")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

subsection("Basic LCEL Chain")
chain = (
    PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    | llm
    | StrOutputParser()
)
print(chain.invoke({"product": "colorful socks"}))

subsection("Restaurant Name Chain")
name_chain = (
    PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )
    | llm
    | StrOutputParser()
)
print(name_chain.invoke({"cuisine": "Mexican"}))

subsection("Sequential Chain: cuisine → name → menu")
llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

name_step = PromptTemplate.from_template(
    "I want to open a restaurant for {cuisine} food. Suggest a fancy name."
) | llm2 | StrOutputParser()

menu_step = PromptTemplate.from_template(
    "Suggest some menu items for {restaurant_name}."
) | llm2 | StrOutputParser()

full_chain = (
    {"restaurant_name": name_step, "cuisine": RunnablePassthrough()}
    | RunnablePassthrough.assign(menu_items=menu_step)
)

result = full_chain.invoke({"cuisine": "South Indian"})
print("Restaurant Name:", result["restaurant_name"])
print("Menu Items:",      result["menu_items"])


# ══════════════════════════════════════════════════════════════════════════
# 06: AGENTS AND TOOLS
# ══════════════════════════════════════════════════════════════════════════
section("06: AGENTS AND TOOLS (langgraph create_react_agent)")

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

llm_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = DuckDuckGoSearchRun()
python_repl = PythonREPLTool()

subsection("Agent with DuckDuckGo Search")
agent = create_react_agent(llm_agent, [search_tool])
response = agent.invoke({"messages": [("user", "What was the GDP of the United States in 2023?")]})
print(response["messages"][-1].content)

subsection("Agent with DuckDuckGo Search + Python REPL")
agent2 = create_react_agent(llm_agent, [search_tool, python_repl])
response2 = agent2.invoke({"messages": [("user", "What is 25 * 17? Also, who invented the telephone?")]})
print(response2["messages"][-1].content)


# ══════════════════════════════════════════════════════════════════════════
# 07: MEMORY / CONVERSATION HISTORY
# ══════════════════════════════════════════════════════════════════════════
section("07: MEMORY / CONVERSATION HISTORY")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

llm_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain_mem = chat_prompt | llm_chat | StrOutputParser()

subsection("Buffer Memory — Full Conversation History")
store: dict = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain_mem,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

session_cfg = {"configurable": {"session_id": "user-1"}}

for q in [
    "Who won the first cricket World Cup?",
    "How much is 5 + 5?",
    "Who was the captain of the winning team?",
]:
    ans = chain_with_history.invoke({"input": q}, config=session_cfg)
    print(f"  Q: {q}")
    print(f"  A: {ans}\n")

print("  --- Full History ---")
for msg in store["user-1"].messages:
    print(f"  {msg.type.upper()}: {msg.content}")

subsection("Window Memory — Keep Last K Turns")

class WindowedChatMessageHistory(ChatMessageHistory):
    k: int = 6

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k:]

store_win: dict = {}

def get_windowed_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_win:
        store_win[session_id] = WindowedChatMessageHistory(k=6)
    return store_win[session_id]

chain_windowed = RunnableWithMessageHistory(
    chain_mem,
    get_windowed_history,
    input_messages_key="input",
    history_messages_key="history",
)

win_cfg = {"configurable": {"session_id": "window-session"}}

for q in [
    "Who won the first cricket World Cup?",
    "How much is 5 + 5?",
    "Who was the captain of the winning team?",
]:
    ans = chain_windowed.invoke({"input": q}, config=win_cfg)
    print(f"  Q: {q}")
    print(f"  A: {ans}\n")

print(f"  Stored messages: {len(store_win['window-session'].messages)}")


# ══════════════════════════════════════════════════════════════════════════
# 08: DOCUMENT LOADERS + RAG
# ══════════════════════════════════════════════════════════════════════════
section("08: DOCUMENT LOADERS + RAG")

import fitz
import base64
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

PDF_PATH = r"D:\AI\AI_Training_Workspace\04_Langchain\PDF\01_AFF_Acceptable_Use_Policy_2025.pdf"

subsection("Extracting PDF Text via GPT-4o Vision (Windows — no poppler/tesseract needed)")

vision_llm = ChatOpenAI(model="gpt-4o-mini")
doc = fitz.open(PDF_PATH)
print(f"  Total pages found: {len(doc)}")

pages = []
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=200)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    response = vision_llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": "Extract all text from this document page exactly as it appears."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ])
    ])
    page_text = response.content
    print(f"  Page {i+1}: {len(page_text)} characters extracted via vision")
    if page_text.strip():
        pages.append(Document(
            page_content=page_text,
            metadata={"page": i, "source": PDF_PATH}
        ))

doc.close()
print(f"\n  Total pages with text: {len(pages)}")
if pages:
    print(f"\n  --- Page 1 preview ---")
    print(f"  {pages[0].page_content[:500]}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
print(f"\n  Number of chunks: {len(splits)}")

subsection("RAG Pipeline — FAISS Vector Store + QA")
if not splits:
    print("  Could not extract any text from PDF.")
else:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

    rag_prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context below.\n\n"
        "Context: {context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

    answer = rag_chain.invoke("What is this document about?")
    print(f"  Q: What is this document about?")
    print(f"  A: {answer}")

section("ALL SECTIONS COMPLETED SUCCESSFULLY")