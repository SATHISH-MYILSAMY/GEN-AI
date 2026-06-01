"""
Multi-DataFrame Agents with LangChain
======================================
Uses the latest LangChain / LangChain-OpenAI APIs:
  - langchain_openai.ChatOpenAI  (replaces deprecated langchain.llms.OpenAI)
  - agent.invoke({"input": ...})  (replaces deprecated agent.run())
"""

import os
import warnings
warnings.filterwarnings("ignore")

import logging
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"   # replace with your key

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════
# 01: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
section("01: LOAD DATA")

url = "https://raw.githubusercontent.com/adamerose/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(f"  Dataset shape: {df.shape}")
print(f"\n  First 5 rows:")
print(df.head().to_string(index=True))


# ══════════════════════════════════════════════════════════════════════════
# 02: SETUP LLM
# ══════════════════════════════════════════════════════════════════════════
section("02: SETUP LLM")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",   # or "gpt-4o" / "gpt-4-turbo"
    temperature=0,
)
print("  Model: gpt-3.5-turbo | Temperature: 0")


# ══════════════════════════════════════════════════════════════════════════
# 03: SINGLE DATAFRAME AGENT
# ══════════════════════════════════════════════════════════════════════════
section("03: SINGLE DATAFRAME AGENT")

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True, # It will show the entire execution details.
    allow_dangerous_code=True, # Allows the agent to execute code that may be potentially harmful. Use with caution
    agent_type="openai-tools",
)

subsection("Query 1 — Row Count")
result1 = agent.invoke({"input": "How many rows are there?"})
print(f"  Answer: {result1['output']}")

subsection("Query 2 — Age Filter")
result2 = agent.invoke({"input": "How many people have age greater than 23?"})
print(f"  Answer: {result2['output']}")


# ══════════════════════════════════════════════════════════════════════════
# 04: MULTI-DATAFRAME AGENT — Two DataFrames
# ══════════════════════════════════════════════════════════════════════════
section("04: MULTI-DATAFRAME AGENT — df vs df1 (age NaN filled)")

df1 = df.copy()
df1["age"] = df1["age"].fillna(df1["age"].mean())

print(f"  df  — NaN in age column : {df['age'].isna().sum()}")
print(f"  df1 — NaN in age column : {df1['age'].isna().sum()} (filled with mean)")

agent_two = create_pandas_dataframe_agent(
    llm,
    [df, df1],
    verbose=True, 
    allow_dangerous_code=True, 
    agent_type="openai-tools",
)

subsection("Query — Difference in age column between df and df1")
result3 = agent_two.invoke({"input": "How many rows in the age column are different between the two dataframes?"})
print(f"  Answer: {result3['output']}")


# ══════════════════════════════════════════════════════════════════════════
# 05: MULTI-DATAFRAME AGENT — Three DataFrames
# ══════════════════════════════════════════════════════════════════════════
section("05: MULTI-DATAFRAME AGENT — df, df1, df2 (Age_Multiplied column added)")

df2 = df1.copy()
df2["Age_Multiplied"] = df1["age"] * 2

print(f"  df  columns ({len(df.columns)}):  {list(df.columns)}")
print(f"  df1 columns ({len(df1.columns)}): {list(df1.columns)}")
print(f"  df2 columns ({len(df2.columns)}): {list(df2.columns)}")
print(f"\n  df2 first 5 rows:")
print(df2.head().to_string(index=True))

agent_three = create_pandas_dataframe_agent(
    llm,
    [df, df1, df2],
    verbose=True,
    allow_dangerous_code=True,
    agent_type="openai-tools",
)

subsection("Query — Column count comparison across all three DataFrames")
result4 = agent_three.invoke({"input": "Are the number of columns the same in all three dataframes?"})
print(f"  Answer: {result4['output']}")


section("ALL SECTIONS COMPLETED SUCCESSFULLY")