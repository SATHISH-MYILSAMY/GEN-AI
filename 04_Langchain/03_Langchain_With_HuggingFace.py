"""
LangChain with Hugging Face
============================
Modern LangChain v0.3+ patterns using:
  - huggingface_hub.InferenceClient   (replaces deprecated HuggingFaceHub)
  - langchain_huggingface.HuggingFacePipeline  (replaces langchain.llms.HuggingFacePipeline)
  - LCEL pipe operator |              (replaces deprecated LLMChain + chain.run())
  - BitsAndBytesConfig                (replaces deprecated load_in_8bit=True)

  NOTE on model compatibility:
    The new HuggingFace Inference API routes through inference providers.
    Only instruction-tuned / chat models work with chat_completion.
    Confirmed working free models:
      - meta-llama/Llama-3.2-1B-Instruct   (lightweight, fast)
      - meta-llama/Llama-3.2-3B-Instruct   (slightly larger)
      - Qwen/Qwen2.5-72B-Instruct           (powerful, free tier)

Install:
  pip install langchain langchain_community langchain_huggingface
  pip install huggingface_hub transformers accelerate bitsandbytes
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_hub_api_token_here"   # https://huggingface.co/settings/tokens

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════
# 02: IMPORTS
# ══════════════════════════════════════════════════════════════════════════
section("02: IMPORTS")

from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 03: ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════
section("03: ENVIRONMENT")

print(f"  HuggingFace token set: {'Yes' if os.environ.get('HUGGINGFACEHUB_API_TOKEN') else 'No'}")


# ══════════════════════════════════════════════════════════════════════════
# 04: APPROACH 1 — HuggingFace Inference API (No Download)
# ══════════════════════════════════════════════════════════════════════════
section("04: APPROACH 1 — HuggingFace Inference API (No local download needed)")

def make_hf_chain(model_id: str, max_tokens: int = 128,
                  temperature: float = 0.1, system_prompt: str = None):
    """
    LCEL-compatible chain using HuggingFace InferenceClient.chat_completion.
    Only works with instruction-tuned / chat-compatible models.
    """
    client = InferenceClient(
        model=model_id,
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )

    def hf_invoke(prompt_value) -> str:
        text = prompt_value.text if hasattr(prompt_value, "text") else str(prompt_value)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    return RunnableLambda(hf_invoke)


# ── 04a: Llama-3.2-1B-Instruct — lightweight, replaces flan-t5 ───────────
subsection("04a: meta-llama/Llama-3.2-1B-Instruct — Seq2Seq replacement (lightweight)")
print("  Model: meta-llama/Llama-3.2-1B-Instruct")

llama_1b = make_hf_chain(
    "meta-llama/Llama-3.2-1B-Instruct",
    max_tokens=64,
    temperature=0.1,
)

chain1 = (
    PromptTemplate.from_template("What is a good name for a company that makes {product}? Give one short name only.")
    | llama_1b
    | StrOutputParser()
)
print(f"  Company name (colorful socks): {chain1.invoke({'product': 'colorful socks'}).strip()}")

chain2 = (
    PromptTemplate.from_template("In one sentence, tell me about famous footballer {name}.")
    | llama_1b
    | StrOutputParser()
)
print(f"  About Messi                  : {chain2.invoke({'name': 'Messi'}).strip()}")

chain3 = (
    PromptTemplate.from_template("List 3 popular food items for a {cuisine} restaurant.")
    | llama_1b
    | StrOutputParser()
)
print(f"  Indian restaurant food items : {chain3.invoke({'cuisine': 'indian'}).strip()}")


# ── 04b: Llama-3.2-1B-Instruct — used again for decoder-only demo ────────
subsection("04b: meta-llama/Llama-3.2-1B-Instruct — Decoder-Only style queries (replaces falcon-7b)")
print("  Model: meta-llama/Llama-3.2-1B-Instruct")
print("  Note: Using same model as 04a — Llama-3.2-3B requires provider access in HF settings")

llama_3b = make_hf_chain(
    "meta-llama/Llama-3.2-1B-Instruct",   # ← same as 04a, confirmed working
    max_tokens=128,
    temperature=0.3,                        # slightly higher temp for variety
)

chain4 = (
    PromptTemplate.from_template("Can you tell me about famous footballer {name}?")
    | llama_3b
    | StrOutputParser()
)
print(f"  About Messi          : {chain4.invoke({'name': 'Messi'}).strip()}")

chain5 = (
    PromptTemplate.from_template("What is a good name for a company that makes {product}? Give one short name only.")
    | llama_3b
    | StrOutputParser()
)
print(f"  Company name         : {chain5.invoke({'product': 'colorful socks'}).strip()}")

chain6 = (
    PromptTemplate.from_template("List 3 popular food items for a {cuisine} restaurant.")
    | llama_3b
    | StrOutputParser()
)
print(f"  Indian food items    : {chain6.invoke({'cuisine': 'indian'}).strip()}")

# ══════════════════════════════════════════════════════════════════════════
# 05: APPROACH 2 — (HuggingFace Pipeline)
# ══════════════════════════════════════════════════════════════════════════
section("05: APPROACH 2 — (HuggingFace Pipeline)")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline   # ✅ replaces langchain.llms.HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model_id = "google/flan-t5-large"

subsection("05a: Load Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=True)
print(f"  Tokenizer loaded: {model_id}")

subsection("05b: Load Model with 8-bit Quantization")
# ✅ BitsAndBytesConfig replaces deprecated load_in_8bit=True
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
print("  Model loaded with 8-bit quantization")

subsection("05c: Create Pipeline and Wrap as LangChain LLM")
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=128)
local_llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("  HuggingFacePipeline LLM created")

subsection("05d: Query 1 — Company Name")
chain = (
    PromptTemplate.from_template("What is a good name for a company that makes {product}")
    | local_llm
    | StrOutputParser()
)
print(f"  Company name: {chain.invoke({'product': 'colorful socks'})}")

subsection("05e: Query 2 — Famous Footballer")
chain2 = (
    PromptTemplate.from_template("Can you tell me about famous footballer {name}")
    | local_llm
    | StrOutputParser()
)
print(f"  About Messi: {chain2.invoke({'name': 'Messi'})}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")