# vars.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agno.models.groq.groq import Groq

load_dotenv()


# System Prompts
SYSTEM_PROMPT_FOR_SUBQUESTIONS = """
You are an expert research planner.
"""

# LLM Configuration
# Consider models known for strong reasoning and instruction following.
# If using Groq, Llama 3 70b or Mixtral are good choices.
# If using Ollama, ensure you have a powerful model downloaded.
ENABLE_LOCAL = False # Set to True to use Ollama, False for Groq/Cloud LLMs

# --- Model IDs ---
# Model for general conversation, synthesis, and potentially grading if capable
# Llama 3 70b on Groq is a strong candidate
REMOTE_LLM = "meta-llama/llama-4-scout-17b-16e-instruct"
LOCAL_LLM = "llama3.2:latest" # Example if using Ollama

# Model specifically for generating subquestions and deciding on decomposition
# Needs good reasoning capabilities.
REASONING_LLM_REMOTE = "meta-llama/llama-4-scout-17b-16e-instruct" # Or Mixtral 8x7b
REASONING_LLM_LOCAL = "smallthinker:latest" # Or a fine-tuned reasoning model

# Model specifically for reliable tool calling (if different from REMOTE_LLM)
# Some models are better fine-tuned for function/tool calling.
TOOL_CALL_LLM_REMOTE = "meta-llama/llama-4-scout-17b-16e-instruct" # Often the main model works well if prompted correctly
TOOL_CALL_LLM_LOCAL = "llama3.2:latest"

# --- Deep Research Parameters ---
# Increased limits for potentially more thorough research
MAX_SEARCH_CALLS = 5 # Max Tavily/Tool calls *within* deep research recursion
MAX_DEPTH = 2        # Max recursion depth for subquestions
NUM_SUBQUESTIONS = 3 # Initial number of subquestions

# --- Knowledge Base ---
# Add path to your vector store if needed, or configure as necessary
VECTOR_STORE_PATH = "../db/chroma.sqlite3" 

# --- API Keys ---
# Ensure these are set in your .env file
# GROQ_API_KEY=your_groq_key
# TAVILY_API_KEY=your_tavily_key
# OPENAI_API_KEY=your_openai_key (if using OpenAI models)

# --- Helper Function to Get Model ID based on ENABLE_LOCAL ---
def get_llm_id(type="remote"):
    if ENABLE_LOCAL:
        if type == "reasoning":
            return REASONING_LLM_LOCAL
        elif type == "tool":
            return TOOL_CALL_LLM_LOCAL
        else:
            return LOCAL_LLM
    else:
        if type == "reasoning":
            return REASONING_LLM_REMOTE
        elif type == "tool":
            return TOOL_CALL_LLM_REMOTE
        else:
            return REMOTE_LLM

def get_llm_provider(model_id, framework="agno"):
    """Returns the appropriate Langchain/Agno LLM class based on config"""
    if ENABLE_LOCAL:
        if framework == "agno":
            from agno.models.ollama import Ollama
            # from langchain_community.chat_models import ChatOllama # Or use Langchain's Ollama
            print(f"Using local Ollama model: {model_id}")
            return Ollama(id=model_id) # Adjust parameters as needed (temp, format)
        # return ChatOllama(model=model_id, temperature=0)
        if framework == "langchain":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=model_id, temperature=0)
    else:
        if framework == "agno":
            # Assuming Groq for remote, add others (Gemini, OpenAI) if needed
            import os
            print(f"Using remote Groq model: {model_id}")
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            return Groq(id=model_id, temperature=0)
        if framework == "langchain":
            return ChatGroq(model=model_id, temperature=0)