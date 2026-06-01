"""
Mistral 7B with LangChain
===========================
Modern LangChain v0.3+ patterns using:
  - transformers.pipeline + langchain_huggingface.HuggingFacePipeline
  - langchain_core.prompts.PromptTemplate   (replaces langchain.prompts.PromptTemplate)
  - LCEL pipe operator |                    (replaces deprecated LLMChain + chain.run())

NOTE: Mistral-7B-v0.1 is a gated repo — requires:
  - A HuggingFace account with access approved at:
    https://huggingface.co/mistralai/Mistral-7B-v0.1
  - Login via: hf auth login   (before executing this script)
  - A GPU with at least 14GB VRAM recommended for bfloat16 inference

Install:
  pip install langchain langchain_huggingface transformers accelerate bitsandbytes torch sentencepiece
"""

import warnings
import logging
warnings.filterwarnings("ignore")                                          # suppresses all non-critical warnings
logging.getLogger("transformers").setLevel(logging.ERROR)                  # suppresses transformers library warnings
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"                     # suppresses advisory warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"]            = "false"                  # suppresses parallelism warnings from tokenizers

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

import torch                                                               # required for bfloat16 dtype and device_map support
import transformers                                                        # provides the pipeline API for text generation
from transformers import AutoTokenizer                                     # tokenizer class that auto-selects based on model name
from langchain_huggingface import HuggingFacePipeline                     # ✅ replaces deprecated langchain.llms.HuggingFacePipeline
from langchain_core.prompts import PromptTemplate                          # ✅ replaces langchain.prompts.PromptTemplate
from langchain_core.output_parsers import StrOutputParser                  # parses LLM output as plain string

print("  All libraries imported successfully.")


# ══════════════════════════════════════════════════════════════════════════
# 02: MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
section("02: MODEL CONFIGURATION")

model = "mistralai/Mistral-7B-v0.1"                                        # Mistral 7B base model — gated, requires HuggingFace access approval
# model = "mistralai/Mistral-7B-Instruct-v0.2"                            # Alternative: instruction-tuned variant (also gated)

print(f"  Model : {model}")
print(f"  Note  : Requires HuggingFace login and Mistral access approval")
print(f"  Visit : https://huggingface.co/mistralai/Mistral-7B-v0.1")
print(f"  Run   : hf auth login   (before executing this script)")


# ══════════════════════════════════════════════════════════════════════════
# 03: LOAD TOKENIZER
# ══════════════════════════════════════════════════════════════════════════
section("03: LOAD TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(model)                           # loads the tokenizer for the specified Mistral model
print(f"  Tokenizer loaded for model: {model}")


# ══════════════════════════════════════════════════════════════════════════
# 04: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM
# ══════════════════════════════════════════════════════════════════════════
section("04: CREATE PIPELINE AND WRAP AS LANGCHAIN LLM")

subsection("04a: Create HuggingFace Text Generation Pipeline")
pipeline = transformers.pipeline(
    "text-generation",                                                     # task type — generates text from a prompt
    model=model,                                                           # model to use for generation
    tokenizer=tokenizer,                                                   # tokenizer to encode/decode text
    torch_dtype=torch.bfloat16,                                           # bfloat16 reduces memory usage with minimal quality loss
    device_map="auto",                                                     # automatically distributes model across available GPUs
    max_new_tokens=512,                                                    # controls output length only (avoids max_length warning)
    do_sample=True,                                                        # enables sampling for more varied responses
    top_k=10,                                                              # restricts sampling to top 10 most probable tokens
    top_p=0.95,                                                            # nucleus sampling — considers tokens covering 95% of probability mass
    num_return_sequences=1,                                                # generate only 1 response per prompt
    eos_token_id=tokenizer.eos_token_id                                    # stop generation at the end-of-sequence token
)
print("  Pipeline created successfully.")

subsection("04b: Wrap Pipeline as LangChain LLM")
llm = HuggingFacePipeline(
    pipeline=pipeline,                                                     # wraps the HuggingFace pipeline as a LangChain-compatible LLM
    model_kwargs={"temperature": 0.7}                                      # temperature=0.7 balances creativity and coherence
)
print("  HuggingFacePipeline LLM created.")


# ══════════════════════════════════════════════════════════════════════════
# 05: DIRECT PROMPTING
# ══════════════════════════════════════════════════════════════════════════
section("05: DIRECT PROMPTING")

subsection("Query 1 — Company Name")
prompt1 = "What would be a good name for a company that makes colorful socks?"
result1 = llm.invoke(prompt1)                                              # ✅ invoke() replaces deprecated llm() call
print(f"  Prompt : {prompt1}")
print(f"  Answer : {result1}")

subsection("Query 2 — Restaurant Name")
prompt2 = "I want to open a restaurant for Indian food. Suggest a fancy name for this."
result2 = llm.invoke(prompt2)
print(f"  Prompt : {prompt2}")
print(f"  Answer : {result2}")


# ══════════════════════════════════════════════════════════════════════════
# 06: PROMPT TEMPLATES (LCEL — replaces LLMChain)
# ══════════════════════════════════════════════════════════════════════════
section("06: PROMPT TEMPLATES (LCEL — replaces LLMChain + chain.run())")

subsection("Example 1 — Restaurant Name from Cuisine")
prompt_template1 = PromptTemplate(
    input_variables=["cuisine"],                                           # 'cuisine' will be filled in by the caller
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)

# ✅ LCEL chain replaces LLMChain(llm=llm, prompt=prompt_template1).run(...)
chain1 = prompt_template1 | llm | StrOutputParser()                       # chains prompt → LLM → string parser using LCEL pipe operator

input_prompt1 = prompt_template1.format(cuisine="Mexican")                # formats the template with 'Mexican' as cuisine
print(f"  Formatted prompt : {input_prompt1}")

result1 = chain1.invoke({"cuisine": "Mexican"})                           # ✅ invoke() replaces deprecated chain.run()
print(f"  Answer           : {result1}")

subsection("Example 2 — Intelligent Chatbot Q&A")
prompt_template2 = PromptTemplate(
    input_variables=["question"],                                          # 'question' will be filled in by the caller
    template=(
        "You are an intelligent assistant. Answer the following question clearly and concisely.\n"
        "Question: {question}\n"
        "Answer:"
    )
)

# ✅ LCEL chain replaces LLMChain(llm=llm, prompt=prompt_template2).run(...)
chain2 = prompt_template2 | llm | StrOutputParser()                       # chains prompt → LLM → string parser using LCEL pipe operator

input_prompt2 = prompt_template2.format(question="What is the Mistral 7B model?")
print(f"  Formatted prompt : {input_prompt2}")

result2 = chain2.invoke({"question": "What is the Mistral 7B model?"})    # ✅ invoke() replaces deprecated chain.run()
print(f"  Answer           : {result2}")

subsection("Example 3 — Book Summary")
prompt_template3 = PromptTemplate(
    input_variables=["book_name"],                                         # 'book_name' will be filled in by the caller
    template="Provide me a concise summary of the book {book_name}."
)

chain3 = prompt_template3 | llm | StrOutputParser()                       # chains prompt → LLM → string parser using LCEL pipe operator

input_prompt3 = prompt_template3.format(book_name="The Alchemist")
print(f"  Formatted prompt : {input_prompt3}")

result3 = chain3.invoke({"book_name": "The Alchemist"})                   # ✅ invoke() replaces deprecated chain.run()
print(f"  Answer           : {result3}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")