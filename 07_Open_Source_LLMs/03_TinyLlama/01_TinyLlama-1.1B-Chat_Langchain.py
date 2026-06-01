"""
TinyLlama 1.1B Chat with LangChain
=====================================
Switched from falcon-rw-1b (base model) to TinyLlama-1.1B-Chat-v1.0
(instruction-tuned) for significantly better answer quality on CPU.

Modern LangChain v0.3+ patterns using:
  - transformers.pipeline + langchain_huggingface.HuggingFacePipeline
  - langchain_core.prompts.PromptTemplate
  - LCEL pipe operator |

Why TinyLlama over falcon-rw-1b?
  - falcon-rw-1b  : base model — continues text, not instruction-following
  - TinyLlama-Chat: instruction-tuned — follows prompts, gives focused answers
  - Same size (~1.1B params), same CPU memory footprint (~2GB RAM)

Install:
  pip install langchain langchain_huggingface transformers accelerate torch
"""

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"
os.environ["CLEAN_UP_TOKENIZATION_SPACES"]      = "false"                  # ✅ suppresses BPE tokenizer warning globally

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
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
section("02: MODEL CONFIGURATION")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"                           # ✅ instruction-tuned — much better than falcon-rw-1b

print(f"  Model  : {model_id}")
print(f"  Note   : Publicly available — no HuggingFace access approval required")
print(f"  Type   : Instruction-tuned chat model — follows prompts precisely")
print(f"  Device : CPU (forced — avoids disk-offload error on low-VRAM GPU)")


# ══════════════════════════════════════════════════════════════════════════
# 03: LOAD TOKENIZER
# ══════════════════════════════════════════════════════════════════════════
section("03: LOAD TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token                    = tokenizer.eos_token               # ✅ prevents pad_token warning
tokenizer.clean_up_tokenization_spaces = False                             # ✅ suppresses BPE tokenizer warning

print(f"  Tokenizer loaded for model: {model_id}")


# ══════════════════════════════════════════════════════════════════════════
# 04: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM
# ══════════════════════════════════════════════════════════════════════════
section("04: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM")

subsection("04a: Load Model")

# ✅ Detect and report device
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU detected : {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    print(f"  Loading on   : CPU (forced — insufficient VRAM for this model)")
else:
    print(f"  No GPU detected — loading on CPU")

model_obj = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,                                                   # ✅ float32 — stable on CPU
    device_map="cpu",                                                      # ✅ force CPU — avoids disk offload entirely
    offload_folder="offload_cache",                                        # ✅ safe fallback folder if any offload occurs
)

# ✅ Replace entire generation_config — eliminates stale max_length conflict
model_obj.generation_config = GenerationConfig(
    do_sample=True,
    top_k=50,                                                              # wider sampling — better for chat model
    top_p=0.95,                                                            # nucleus sampling
    temperature=0.7,                                                       # balanced creativity vs coherence
    repetition_penalty=1.1,                                                # light penalty — chat models need less
    max_new_tokens=300,                                                    # ✅ set on model — avoids pipeline conflict
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# ✅ Clear stale max_length baked into model config — stops max_length=20 warning
model_obj.config.max_length = None

print("  Model loaded successfully on CPU.")

subsection("04b: Create HuggingFace Text Generation Pipeline")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    return_full_text=False,                                                # ✅ returns ONLY new tokens — no prompt echo
    device=-1,                                                             # ✅ -1 = CPU — matches device_map above
)

print("  Pipeline created successfully.")

subsection("04c: Wrap Pipeline as LangChain LLM")

llm = HuggingFacePipeline(pipeline=pipeline)                               # ✅ no pipeline_kwargs — config lives on model_obj

print("  HuggingFacePipeline LLM created.")


# ══════════════════════════════════════════════════════════════════════════
# 05: PROMPT TEMPLATE — INTELLIGENT CHATBOT (LCEL)
# ══════════════════════════════════════════════════════════════════════════
section("05: PROMPT TEMPLATE — INTELLIGENT CHATBOT (LCEL)")

# ✅ TinyLlama uses ChatML format — <|system|> and <|user|> tags
# give best instruction-following quality for this model
template = """<|system|>
You are a helpful, accurate and concise assistant. Answer the question clearly and directly.</s>
<|user|>
{question}</s>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

# ✅ Strip leading/trailing whitespace from LLM output
cleaner = RunnableLambda(lambda x: x.strip())

# ✅ LCEL chain: prompt → LLM → string parser → cleaner
chain = prompt | llm | StrOutputParser() | cleaner

print("  Prompt template and LCEL chain created.")

# ✅ Silence remaining inference-time warnings from transformers internals
import transformers as _tf
_original_warn = _tf.utils.logging.get_logger("transformers").warning
_tf.utils.logging.get_logger("transformers").setLevel(logging.CRITICAL)

# ══════════════════════════════════════════════════════════════════════════
# 06: DIAGNOSTIC — RAW PIPELINE TEST
# ══════════════════════════════════════════════════════════════════════════
section("06: DIAGNOSTIC — RAW PIPELINE TEST")

print("  Running raw pipeline test (may take 30-60s on CPU)...")
raw_out = pipeline("<|user|>\nSay hello in one sentence.</s>\n<|assistant|>")
print(f"  Raw pipeline output: {raw_out[0]['generated_text'].strip()}")


# ══════════════════════════════════════════════════════════════════════════
# 07: RUN QUERIES
# ══════════════════════════════════════════════════════════════════════════
section("07: RUN QUERIES")

subsection("Query 1 — AI as Nursery Rhymes")
print("  Running query (may take 30-60s on CPU)...")
question1 = "Explain what is Artificial Intelligence as Nursery Rhymes"
result1 = chain.invoke({"question": question1})
print(f"\n  Answer : {result1}")

subsection("Query 2 — Code for Adding Two Numbers")
print("  Running query (may take 30-60s on CPU)...")
question2 = "Give me Python code for adding 2 numbers"
result2 = chain.invoke({"question": question2})
print(f"\n  Answer : {result2}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")