"""
Llama 2 with LangChain — FIXED (CPU Offload for Low VRAM)
===========================================================
Fix applied for error:
  "Some modules are dispatched on the CPU or the disk.
   Make sure you have enough GPU RAM to fit the quantized model."

Root cause : GPU has insufficient VRAM to hold the 4-bit quantized Llama-2-7B
             (~4–5GB needed). BitsAndBytes refuses to proceed when layers spill
             to CPU without explicit permission.

Fix summary:
  1. Add `llm_int8_enable_fp32_cpu_offload=True` to BitsAndBytesConfig
  2. Build an explicit device_map via infer_auto_device_map() rather than "auto"
  3. Pass device_map inside model_kwargs (not as a top-level pipeline argument)

Modern LangChain v0.3+ patterns:
  - transformers.pipeline + langchain_huggingface.HuggingFacePipeline
  - langchain_core.prompts.PromptTemplate
  - LCEL pipe operator |  (replaces deprecated LLMChain + chain.run())

Install:
  pip install langchain langchain_huggingface transformers accelerate bitsandbytes torch
  hf auth login   # login before running
"""

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"

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
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from accelerate import infer_auto_device_map                               # ✅ NEW: needed to build explicit device_map
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
section("02: MODEL CONFIGURATION")

model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "daryl149/llama-2-7b-chat-hf"   # Alternative: no approval needed

print(f"  Model : {model_id}")
print(f"  Note  : Requires HuggingFace login and Llama 2 access approval")
print(f"  Run   : hf auth login   (before executing this script)")


# ══════════════════════════════════════════════════════════════════════════
# 03: GPU DIAGNOSTICS  ← NEW SECTION
# ══════════════════════════════════════════════════════════════════════════
section("03: GPU DIAGNOSTICS")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {free/1e9:.1f} GB free  /  {total/1e9:.1f} GB total")
    # Use 80% of free VRAM to leave headroom for activations
    free_vram_gb  = torch.cuda.mem_get_info(0)[0] / 1e9
    gpu_alloc_gb  = max(1, int(free_vram_gb * 0.8))
    print(f"  Allocating {gpu_alloc_gb} GB to GPU 0 for model layers")
else:
    gpu_alloc_gb = 0
    print("  ⚠  No CUDA GPU detected — model will run entirely on CPU (slow)")

cpu_alloc_gb = 24                                                          # adjust to your system RAM
print(f"  CPU RAM budget : {cpu_alloc_gb} GB")


# ══════════════════════════════════════════════════════════════════════════
# 04: LOAD TOKENIZER
# ══════════════════════════════════════════════════════════════════════════
section("04: LOAD TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"  Tokenizer loaded for model: {model_id}")


# ══════════════════════════════════════════════════════════════════════════
# 05: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM
# ══════════════════════════════════════════════════════════════════════════
section("05: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM")

# ── 05a: 4-bit Quantization Config ───────────────────────────────────────
subsection("05a: 4-bit Quantization Config")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,                                 # ✅ FIX 1: allows layers to offload to CPU in fp32
)
print("  4-bit quantization config created (CPU offload enabled).")

# ── 05b: Build Explicit device_map ───────────────────────────────────────
subsection("05b: Build Explicit device_map")

# infer_auto_device_map inspects each layer's size and assigns it to GPU or
# CPU based on the memory budgets you provide.  This replaces device_map="auto"
# which was silently trying to put everything on GPU and failing.
max_memory = {}
if gpu_alloc_gb > 0:
    max_memory[0]     = f"{gpu_alloc_gb}GiB"                              # GPU budget
max_memory["cpu"]     = f"{cpu_alloc_gb}GiB"                              # CPU RAM budget

config     = AutoConfig.from_pretrained(model_id)
dummy      = AutoModelForCausalLM.from_config(config)                     # empty skeleton — no weights loaded

device_map = infer_auto_device_map(
    dummy,
    max_memory=max_memory,
    no_split_module_classes=["LlamaDecoderLayer"],                         # keeps each transformer block on one device
)
del dummy                                                                  # free the skeleton
print(f"  device_map built: {dict(list(device_map.items())[:5])} ...")    # preview first 5 entries

# ── 05c: Create HuggingFace Text Generation Pipeline ─────────────────────
subsection("05c: Create HuggingFace Text Generation Pipeline")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={                                                          # ✅ FIX 2: pass both quantization AND device_map here
        "quantization_config": bnb_config,
        "device_map": device_map,                                          # ✅ FIX 3: explicit map — NOT device_map="auto" at top level
    },
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
print("  Pipeline created successfully.")

# ── 05d: Wrap as LangChain LLM ───────────────────────────────────────────
subsection("05d: Wrap Pipeline as LangChain LLM")
llm = HuggingFacePipeline(pipeline=pipeline)
print("  HuggingFacePipeline LLM created.")


# ══════════════════════════════════════════════════════════════════════════
# 06: DIRECT PROMPTING
# ══════════════════════════════════════════════════════════════════════════
section("06: DIRECT PROMPTING")

subsection("Query 1 — Company Name")
prompt1 = "What would be a good name for a company that makes colorful socks?"
result1 = llm.invoke(prompt1)
print(f"  Prompt : {prompt1}")
print(f"  Answer : {result1}")

subsection("Query 2 — Restaurant Name")
prompt2 = "I want to open a restaurant for Indian food. Suggest a fancy name for this."
result2 = llm.invoke(prompt2)
print(f"  Prompt : {prompt2}")
print(f"  Answer : {result2}")


# ══════════════════════════════════════════════════════════════════════════
# 07: PROMPT TEMPLATES (LCEL — replaces LLMChain)
# ══════════════════════════════════════════════════════════════════════════
section("07: PROMPT TEMPLATES (LCEL — replaces LLMChain + chain.run())")

subsection("Example 1 — Restaurant Name from Cuisine")
prompt_template1 = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)
chain1 = prompt_template1 | llm | StrOutputParser()

input_prompt1 = prompt_template1.format(cuisine="Indian")
print(f"  Formatted prompt : {input_prompt1}")
result1 = chain1.invoke({"cuisine": "Indian"})
print(f"  Answer           : {result1}")

subsection("Example 2 — Book Summary")
prompt_template2 = PromptTemplate(
    input_variables=["book_name"],
    template="Provide me a concise summary of the book {book_name}."
)
chain2 = prompt_template2 | llm | StrOutputParser()

input_prompt2 = prompt_template2.format(book_name="The Alchemist")
print(f"  Formatted prompt : {input_prompt2}")
result2 = chain2.invoke({"book_name": "Harry Potter"})
print(f"  Answer           : {result2}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")