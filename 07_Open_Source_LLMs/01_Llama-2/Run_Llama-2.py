"""
How to Run Llama 2 Locally
============================
Runs Llama 2 using llama-cpp-python with GGUF format models.

Key updates from original notebook:
  - GGML format (.bin) is DEPRECATED — replaced with GGUF format (.gguf)
  - TheBloke/Llama-2-13B-chat-GGML is deprecated — use TheBloke/Llama-2-13B-chat-GGUF
  - llama-cpp-python==0.1.78 is outdated — use latest version (>=0.2.0)
  - numpy==1.23.4 pinning removed — use latest compatible numpy
  - lcpp_llm.params removed — use .model_params and .context_params instead

Install (choose one based on your hardware):

  CPU only:
    pip install llama-cpp-python huggingface_hub

  GPU (NVIDIA CUDA):
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python huggingface_hub

  GPU (Apple Silicon):
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python huggingface_hub

Model:
  TheBloke/Llama-2-13B-chat-GGUF on HuggingFace
  Requires accepting Meta's license at: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
"""

import warnings
warnings.filterwarnings("ignore")                                          # suppresses all non-critical warnings

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════
# 01: MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
section("01: MODEL CONFIGURATION")

# ✅ GGML (.bin) is deprecated — GGUF (.gguf) is the current standard format
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"                    # HuggingFace repo with GGUF format models (replaces deprecated GGML repo)
model_basename     = "llama-2-13b-chat.Q5_K_M.gguf"                      # Q5_K_M: good balance of quality and speed (replaces ggmlv3.q5_1.bin)

print(f"  Model repo   : {model_name_or_path}")
print(f"  Model file   : {model_basename}")
print(f"  Format       : GGUF (replaces deprecated GGML .bin format)")


# ══════════════════════════════════════════════════════════════════════════
# 02: IMPORTS
# ══════════════════════════════════════════════════════════════════════════
section("02: IMPORTS")

from huggingface_hub import hf_hub_download                               # downloads model files from HuggingFace Hub
from llama_cpp import Llama                                                # llama-cpp-python wrapper for running GGUF models locally

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 03: DOWNLOAD THE MODEL
# ══════════════════════════════════════════════════════════════════════════
section("03: DOWNLOAD THE MODEL")

print(f"  Downloading '{model_basename}' from '{model_name_or_path}'...")
print(f"  Note: First download may take several minutes (~9GB model)")

model_path = hf_hub_download(
    repo_id=model_name_or_path,                                           # HuggingFace repo containing the GGUF model file
    filename=model_basename                                                # specific GGUF file to download
)

print(f"  Model downloaded to: {model_path}")                            # local cache path where the model file is stored


# ══════════════════════════════════════════════════════════════════════════
# 04: LOAD THE MODEL
# ══════════════════════════════════════════════════════════════════════════
section("04: LOAD THE MODEL")

subsection("Option A — GPU (NVIDIA CUDA)")
print("  Using GPU acceleration with n_gpu_layers=32")

lcpp_llm = Llama(
    model_path=model_path,                                                 # path to the downloaded GGUF model file
    n_threads=2,                                                           # number of CPU threads used for computation
    n_batch=512,                                                           # number of tokens processed in parallel (between 1 and n_ctx)
    n_gpu_layers=32,                                                       # number of model layers offloaded to GPU VRAM (0 = CPU only)
    n_ctx=4096,                                                            # context window size in tokens (max input + output length)
    verbose=False                                                          # suppresses llama.cpp internal logging output
)

print(f"  Model loaded successfully.")
print(f"  GPU layers    : {lcpp_llm.model_params.n_gpu_layers}")         # ✅ .params removed in newer llama-cpp-python — use .model_params
print(f"  Context size  : {lcpp_llm.context_params.n_ctx}")              # ✅ context params now accessed via .context_params
print(f"  Model path    : {lcpp_llm.model_path}")                         # ✅ model_path is already a str in newer llama-cpp-python

# ── CPU only (uncomment if no GPU available) ──────────────────────────────
# subsection("Option B — CPU only")
# lcpp_llm = Llama(
#     model_path=model_path,
#     n_threads=4,                                                         # use more threads for faster CPU inference
#     n_batch=256,                                                         # smaller batch for CPU
#     n_gpu_layers=0,                                                      # 0 = CPU only, no GPU offloading
#     n_ctx=4096,
#     verbose=False
# )


# ══════════════════════════════════════════════════════════════════════════
# 05: CREATE PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════
section("05: CREATE PROMPT TEMPLATE")

prompt = "Write a linear regression code in Python"                       # user query to send to the model

# Llama 2 chat uses a specific system/user/assistant prompt format
prompt_template = f"""SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.

USER: {prompt}

ASSISTANT:
"""

print(f"  Prompt       : {prompt}")
print(f"  Full template:\n{prompt_template}")


# ══════════════════════════════════════════════════════════════════════════
# 06: GENERATE RESPONSE
# ══════════════════════════════════════════════════════════════════════════
section("06: GENERATE RESPONSE")

print("  Generating response (this may take a moment)...")

response = lcpp_llm(
    prompt=prompt_template,                                                # the formatted prompt including system message and user query
    max_tokens=512,                                                        # ✅ reduced from 256 — fewer tokens = faster response
    temperature=0.1,                                                       # ✅ lowered from 0.5 — near-deterministic, skips expensive sampling
    top_p=0.9,                                                             # ✅ slightly tighter nucleus — less candidate evaluation
    repeat_penalty=1.1,                                                    # ✅ reduced from 1.2 — lighter penalty computation
    top_k=40,                                                              # ✅ reduced from 150 — fewer candidates sampled per token = faster
    echo=False                                                             # ✅ disabled — skips re-encoding the prompt in output
)


# ══════════════════════════════════════════════════════════════════════════
# 07: DISPLAY RESPONSE
# ══════════════════════════════════════════════════════════════════════════
section("07: DISPLAY RESPONSE")

subsection("Raw response object")
print(response)                                                            # full response dict including usage stats, finish reason, and choices

subsection("Generated text only")
print(response["choices"][0]["text"])                                      # extracts just the generated text from the response


section("ALL SECTIONS COMPLETED SUCCESSFULLY")