"""
Mistral-7B-Instruct-v0.3 (4-bit Quantized) with LangChain
============================================================
Uses unsloth/mistral-7b-instruct-v0.3-bnb-4bit
  - Public repo   : No HuggingFace token or approval required
  - 4-bit quant   : ~4.5 GB RAM (vs ~14.5 GB for float32 full precision)
  - Same format   : Uses [INST]...[/INST] prompt format

Speed fixes applied:
  - max_new_tokens reduced to 50 — fastest visible output on CPU
  - model.config.max_length cleared — fixes max_new_tokens vs max_length warning
  - do_sample=False, greedy decoding — faster than sampling on CPU

Install:
  pip install langchain langchain_huggingface transformers accelerate torch sentencepiece bitsandbytes
"""

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
section("02: MODEL CONFIGURATION")

model_id = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

print(f"  Model     : {model_id}")
print(f"  Note      : Public, no token — 4-bit quantized Mistral-7B-Instruct-v0.3")
print(f"  RAM usage : ~4.5 GB (vs ~14.5 GB full precision)")
print(f"  Format    : Uses [INST]...[/INST] prompt format")
print(f"  Speed fix : max_new_tokens=50, greedy decoding — fastest CPU output")


# ══════════════════════════════════════════════════════════════════════════
# 03: LOAD TOKENIZER
# ══════════════════════════════════════════════════════════════════════════
section("03: LOAD TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token                    = tokenizer.eos_token
tokenizer.clean_up_tokenization_spaces = False

print(f"  Tokenizer loaded for model: {model_id}")


# ══════════════════════════════════════════════════════════════════════════
# 04: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM
# ══════════════════════════════════════════════════════════════════════════
section("04: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM")

subsection("04a: Configure 4-bit Quantization")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,                                  # ✅ float32 — stable on CPU
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
print("  BitsAndBytesConfig created.")

subsection("04b: Load 4-bit Quantized Model")

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU detected : {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    device_map = "auto"
else:
    print(f"  No GPU detected — loading on CPU")
    device_map = "cpu"

model_obj = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    offload_folder="offload_cache",
    low_cpu_mem_usage=True,
)

# ✅ KEY FIX: clear stale max_length BEFORE setting generation_config
# This eliminates "Both max_new_tokens and max_length seem to have been set" warning
model_obj.config.max_length       = None
model_obj.config.max_new_tokens   = None

# ✅ Greedy decoding (do_sample=False) — fastest on CPU, no sampling overhead
# ✅ max_new_tokens=50 — short answers, much faster response time on CPU
model_obj.generation_config = GenerationConfig(
    do_sample=False,                                                       # ✅ greedy — fastest on CPU
    max_new_tokens=50,                                                     # ✅ short output = fast response
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

print("  Model loaded successfully.")

subsection("04c: Create HuggingFace Text Generation Pipeline")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    return_full_text=False,                                                # ✅ no prompt echo in output
    max_new_tokens=50,                                                     # ✅ must also be set here — overrides any model default
)
print("  Pipeline created successfully.")

subsection("04d: Wrap Pipeline as LangChain LLM")

llm = HuggingFacePipeline(pipeline=pipeline)
print("  HuggingFacePipeline LLM created.")


# ══════════════════════════════════════════════════════════════════════════
# 05: DIRECT PROMPTING
# ══════════════════════════════════════════════════════════════════════════
section("05: DIRECT PROMPTING")

subsection("Query 1 — Company Name")
prompt1 = "[INST] Suggest one short name for a company that makes colorful socks. [/INST]"
result1 = llm.invoke(prompt1)
print(f"  Prompt : Suggest one short name for a company that makes colorful socks.")
print(f"  Answer : {result1.strip()}")

subsection("Query 2 — Restaurant Name")
prompt2 = "[INST] Suggest one fancy name for an Indian food restaurant. [/INST]"
result2 = llm.invoke(prompt2)
print(f"  Prompt : Suggest one fancy name for an Indian food restaurant.")
print(f"  Answer : {result2.strip()}")


# ══════════════════════════════════════════════════════════════════════════
# 06: PROMPT TEMPLATES (LCEL — replaces LLMChain)
# ══════════════════════════════════════════════════════════════════════════
section("06: PROMPT TEMPLATES (LCEL — replaces LLMChain + chain.run())")

cleaner = RunnableLambda(lambda x: x.strip())

subsection("Example 1 — Restaurant Name from Cuisine")

prompt_template1 = PromptTemplate(
    input_variables=["cuisine"],
    template="[INST] Suggest one fancy name for a {cuisine} food restaurant. [/INST]"
)
chain1 = prompt_template1 | llm | StrOutputParser() | cleaner

print(f"  Formatted prompt : {prompt_template1.format(cuisine='Mexican')}")
result1 = chain1.invoke({"cuisine": "Mexican"})
print(f"  Answer           : {result1}")

subsection("Example 2 — Intelligent Chatbot Q&A")

prompt_template2 = PromptTemplate(
    input_variables=["question"],
    template="[INST] Answer in 2 sentences: {question} [/INST]"
)
chain2 = prompt_template2 | llm | StrOutputParser() | cleaner

print(f"  Formatted prompt : {prompt_template2.format(question='What is the Mistral language model?')}")
result2 = chain2.invoke({"question": "What is the Mistral language model?"})
print(f"  Answer           : {result2}")

subsection("Example 3 — Book Summary")

prompt_template3 = PromptTemplate(
    input_variables=["book_name"],
    template="[INST] Summarize the book {book_name} in 2 sentences. [/INST]"
)
chain3 = prompt_template3 | llm | StrOutputParser() | cleaner

print(f"  Formatted prompt : {prompt_template3.format(book_name='The Alchemist')}")
result3 = chain3.invoke({"book_name": "The Alchemist"})
print(f"  Answer           : {result3}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")