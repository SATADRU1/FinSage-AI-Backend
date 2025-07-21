# summarizer.py

# Using Langchain's agent framework as provided, but simplifying the tool part
# as the core task is LLM-based summarization, not complex tool use.
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from termcolor import colored
# from langchain_groq import ChatGroq # Use centralized provider
from vars import get_llm_id, get_llm_provider
from typing import Optional
import os

# Use the main LLM for summarization
summarizer_llm = get_llm_provider(get_llm_id("remote"), framework="langchain")

# Refined prompt for better summarization
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an expert summarization engine. Your goal is to create a concise, accurate, and highly readable summary of the provided text.

    **Instructions:**
    1.  **Identify Key Information:** Extract the most important facts, findings, data points, and conclusions from the text.
    2.  **Synthesize:** Combine related points and eliminate redundancy.
    3.  **Structure:** Organize the summary logically. Use bullet points or numbered lists for clarity if the content lends itself to it (e.g., listing pros/cons, steps, key findings).
    4.  **Accuracy:** Ensure the summary accurately reflects the content and intent of the original text. Do not add external information or opinions.
    5.  **Conciseness:** Be brief but comprehensive. Capture the essence without unnecessary jargon or excessive detail.
    6.  **Readability:** Write in clear, natural language.
    7.  **Format Handling:** If the input format is mentioned (e.g., HTML, JSON), focus on extracting and summarizing the meaningful content within that structure.
    """),
    HumanMessage(content="{text_to_summarize}")
])

# Chain for direct summarization
summarization_chain = prompt_template | summarizer_llm # | StrOutputParser() # Assuming Agno model returns content directly

def summarize(text: str, format_type: Optional[str] = None) -> str:
    """
    Summarizes the provided text using an LLM chain.

    Args:
        text: The text to summarize.
        format_type: Optional hint about the format (currently used to refine the input message).

    Returns:
        A summary of the text.
    """
    if not text or not text.strip():
        return "Error: No text provided to summarize."

    input_text = f"Please summarize the following text"
    if format_type:
        input_text += f" (Note: The text is in {format_type} format)"
    input_text += f":\n\n---\n{text}\n---"

    try:
        response = summarization_chain.invoke({"text_to_summarize": input_text})
        # Assuming response object has a 'content' attribute like Agno/Langchain messages
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
             return response
        else:
             # Handle unexpected response structure
             print(f"Warning: Unexpected summarizer response type: {type(response)}")
             return str(response) # Fallback to string representation
    except Exception as e:
        print(colored(f"Error during summarization: {e}", "red"))
        return f"Error: Could not summarize the text. Details: {str(e)}"


# Note: The previous agent-based summarizer seemed overly complex for this task.
# A direct LLM call with a good prompt is usually sufficient and more reliable.
# If you specifically need the agent structure for other reasons, it can be adapted.