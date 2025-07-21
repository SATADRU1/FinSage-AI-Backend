# retrieval_grader.py

from typing import Dict
from langchain.prompts import PromptTemplate
# Using Langchain's ChatGroq/ChatOllama for consistency if preferred
# from langchain_community.chat_models import ChatOllama
# from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from vars import get_llm_id, get_llm_provider  # Use centralized provider
import os
from dotenv import load_dotenv

load_dotenv()

# LLM for Grading - needs good instruction following and JSON output
# Using the main LLM provider, assuming it's capable.
# Or specify a different one if needed
grading_llm = get_llm_provider(get_llm_id("remote"), "langchain")
# Ensure the provider can handle JSON output mode if available/needed
# e.g., for ChatOllama: llm = ChatOllama(model=..., format="json", temperature=0)
# e.g., for ChatGroq: llm = ChatGroq(model=..., temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
# Note: Agno models might handle JSON output via prompt instructions. Check Agno docs.

# Improved Prompt for Retrieval Grading
prompt_for_retrieval_grading = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 
    1/ a QUESTION
    2/ A FACT provided by the student
    
    You are grading RELEVANCE RECALL:
    A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
    A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION OR/AND the QUESTION is asking for realtime information. 
    1 is the highest (best) score. 0 is the lowest score you can give. 
    
    NOTE that if the question is asking about realtime information, the fact may not be relevant.

    Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
    
    Avoid simply stating the correct answer at the outset.
    
    Question: {question} \n
    Fact: \n\n {documents} \n\n
    
    Give a binary score '1' or '0' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """,
    input_variables=["question", "documents"],
)

retrieval_grader = prompt_for_retrieval_grading | grading_llm | JsonOutputParser()

# --- Small Talk Grader (NEW) ---
# This helps identify queries that don't need complex processing.

prompt_for_small_talk = PromptTemplate(
    template="""You are classifying user input. Determine if the following input is simple small talk, a greeting, a thank you, or a basic conversational phrase, rather than a request for financial information, analysis, or research.

    Examples of Small Talk:
    - "Hello"
    - "How are you?"
    - "Thanks!"
    - "Okay"
    - "Who are you?"
    - "What can you do?"

    Examples of NOT Small Talk (Requires Financial Processing):
    - "What's the price of Apple stock?"
    - "Tell me about Tesla's latest earnings."
    - "Should I invest in Bitcoin?"
    - "Explain diversification."

    Analyze the user input below.

    USER INPUT:
    {question}

    Is this input simple small talk, a greeting, or a basic conversational phrase? Answer with only YES or NO.
    """,
    input_variables=["question"],
)

# Simple Output Parser for YES/NO


class YesNoParser(JsonOutputParser):
    def parse(self, text: str) -> Dict[str, bool]:
        text_upper = text.strip().upper()
        if "YES" in text_upper:
            return {"score": True}
        elif "NO" in text_upper:
            return {"score": False}
        else:
            # Default or raise error - defaulting to False (not small talk) is safer
            print(
                f"Warning: Grader returned ambiguous output: {text}. Defaulting to NO.")
            return {"score": False}


small_talk_grader = prompt_for_small_talk | grading_llm | YesNoParser()
