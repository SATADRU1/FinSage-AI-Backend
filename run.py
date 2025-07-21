# run.py

# Assuming this initializes and returns a Chroma/FAISS etc. vector store
from ingest import vector_store
from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
# Use centralized LLM providers
from vars import (
    get_llm_id, get_llm_provider,
    MAX_SEARCH_CALLS, MAX_DEPTH, VECTOR_STORE_PATH
)
from agno.tools.yfinance import YFinanceTools
# Import graders and summarizer
from retrieval_grader import retrieval_grader, small_talk_grader # Keep existing imports if they define other things
from deep_research import DeepResearch # Import the modified DeepResearch class
# from summarizer import summarize # Not currently used for final synthesis
from tavily import TavilyClient

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.markdown import Markdown
from termcolor import colored
from langchain.memory import ConversationBufferMemory # Import memory
from langchain_core.messages import SystemMessage
from langchain.prompts import PromptTemplate
# Import necessary parsers
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import BooleanOutputParser # Import Boolean parser

# from retrieval_grader import YesNoParser # Remove if SimpleYesNoParser was defined here
from typing import Optional, Callable, Dict, Any # Add Dict, Any, Optional, Callable
import traceback # Import traceback for detailed error logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory # Ensure this is imported
# Add these imports for serving static files
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import os # Make sure os is imported
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory # Ensure this is imported
# Add these imports for serving static files
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import os # Make sure os is imported

load_dotenv()
console = Console()

# --- Initialize Tools ---
tavily_api_key = os.environ.get("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")
tavily_client = TavilyClient(api_key=tavily_api_key)

yf_tool = YFinanceTools(
    stock_price=True,
    analyst_recommendations=True,
    stock_fundamentals=True,
    company_info=True,
)
if not hasattr(yf_tool, 'name'):
    yf_tool.name = "YFinanceTools"
if not hasattr(tavily_client, 'name'):
    tavily_client.name = "TavilySearch"

researcher = DeepResearch(
    max_search_calls=MAX_SEARCH_CALLS, max_depth=MAX_DEPTH)


# --- Initialize Knowledge Base ---
try:
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
except Exception as e:
    print(colored(f"Error initializing vector store/retriever: {e}", "red"))
    print(colored("Knowledge base retrieval will be unavailable.", "yellow"))
    retriever = None

knowledge_base = LangChainKnowledgeBase(
    retriever=retriever) if retriever else None

# --- Initialize LLMs ---
# Ensure framework="langchain" is specified when Langchain specific features like parsers are used
main_llm_langchain = get_llm_provider(get_llm_id("remote"), framework="langchain")
tool_llm_langchain = get_llm_provider(get_llm_id("tool"), framework="langchain")

# Keep Agno-compatible LLMs if needed for Agno Agents
main_llm_agno = get_llm_provider(get_llm_id("remote"))
tool_llm_agno = get_llm_provider(get_llm_id("tool"))


# --- Helper Function for Tool Call Display (Keep if needed) ---
# ... (display_tool_calls function remains the same) ...


# --- Core Processing Function ---
def process_query_flow(
    query: str,
    memory: ConversationBufferMemory,
    deep_search: bool = False,
    stream_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    print(colored(f"\nProcessing Query: '{query}' (Deep Search: {deep_search})", "white", attrs=["bold"]))
    final_answer = ""
    rag_context = ""
    web_research_context = ""
    research_debug_log = ""

    # === 1. Small Talk Check (Using BooleanOutputParser) ===
    try:
        print(colored("Checking for small talk...", "cyan"))
        # Use Langchain compatible LLM
        small_talk_llm = main_llm_langchain
        # Updated prompt for BooleanOutputParser (often works better with true/false but tries yes/no)
        small_talk_prompt = PromptTemplate(
            template="Is the following a simple greeting, pleasantry, or conversational filler (small talk)? Answer ONLY with 'YES' or 'NO'.\n\nQuestion: {question}",
            input_variables=["question"]
        )
        # Use BooleanOutputParser
        small_talk_parser = BooleanOutputParser()
        small_talk_chain = small_talk_prompt | small_talk_llm | small_talk_parser

        # Invoke the chain
        is_small_talk = small_talk_chain.invoke({"question": query})
        print(colored(f"Small talk check result: {is_small_talk}", "magenta"))

        if is_small_talk: # BooleanOutputParser returns True or False
            print(colored("Query identified as small talk.", "yellow"))
            # Use Agno compatible LLM for the Agno Agent
            conv_agent = Agent(
                model=main_llm_agno,
                description="You are a friendly assistant.",
                memory=memory
            )
            history = memory.load_memory_variables({})["chat_history"]
            response = conv_agent.run(f"Respond conversationally to: {query}", chat_history=history)
            return {"answer": response.content, "deep_research_log": ""}
    except Exception as e:
        # Catch potential OutputParserException here too
        print(colored(f"Error during small talk check: {e}", "red"))
        if "Invalid" in str(e) or "OutputParserException" in str(e):
             print(colored("Attempting to proceed assuming it's not small talk...", "yellow"))
        else:
             traceback.print_exc() # Print full trace for unexpected errors
        # Proceed assuming it's not small talk on error


    # === 2. RAG Retrieval ===
    retrieved_docs_content = "No documents found or knowledge base unavailable."
    retrieved_docs = None
    if knowledge_base and retriever:
        try:
            print(colored("Attempting RAG retrieval...", "cyan"))
            retrieved_docs = retriever.invoke(query) # Langchain LCEL standard invoke

            if retrieved_docs:
                retrieved_docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                print(colored(f"Retrieved {len(retrieved_docs)} snippets.", "green"))
                print(colored("Retrieved Snippet:", "yellow"))
                print(colored(retrieved_docs_content[:500] + "...", "yellow"))
            else:
                print(colored("No relevant documents found in knowledge base.", "yellow"))
                retrieved_docs_content = "No relevant documents found in knowledge base."
        except Exception as e:
            print(colored(f"Error during RAG retrieval: {e}", "red"))
            traceback.print_exc()
            retrieved_docs_content = "Error retrieving documents from knowledge base."
    else:
        print(colored("Knowledge base not available, skipping RAG.", "yellow"))

    # === 3. Relevance Grading (Using JsonOutputParser with Markdown Fence Instruction) ===
    grade = 0 # Default to not relevant
    if retrieved_docs:
        try:
            print(colored("Grading retrieved documents...", "cyan"))
            grading_llm = main_llm_langchain
            # Updated prompt asking for JSON within markdown fences
            grading_prompt = PromptTemplate(
                 template="""Evaluate the relevance of the retrieved documents to the user's question. Give a binary score: 1 if relevant, 0 if not.\n
                 Provide the score ONLY as JSON within json markdown code fences. Example:
                 json
                 {{
                   "score": 1
                 }}
                 ```

                 Documents:\n{documents}\n\nQuestion: {question}""",
                 input_variables=["documents", "question"]
            )
            # Standard JsonOutputParser - should handle markdown fences
            grading_parser = JsonOutputParser()
            grading_chain = grading_prompt | grading_llm | grading_parser

            grade_result = grading_chain.invoke({"question": query, "documents": retrieved_docs_content})
            # Check the type/content of grade_result
            print(f"DEBUG: Raw grade_result: {grade_result} (type: {type(grade_result)})")
            if isinstance(grade_result, dict):
                 grade = grade_result.get('score', 0) # Default to 0 if key missing
            else:
                 print(colored("Warning: Grading did not return a dictionary.", "yellow"))
                 grade = 0 # Default to not relevant if parsing failed unexpectedly

            print(colored(f"Retrieval grade: {grade} ({'Relevant' if grade == 1 else 'Not Relevant'})", 'magenta'))

            if grade == 1:
                rag_context = retrieved_docs_content
            else:
                print(colored("Documents deemed not relevant or insufficient.", "yellow"))
                rag_context = ""
        except Exception as e:
            print(colored(f"Error during retrieval grading: {e}", "red"))
            # Don't necessarily need full traceback here if it's the expected OutputParserException
            if "Invalid" in str(e) or "OutputParserException" in str(e):
                 print(colored("Failed to parse relevance grade. Assuming documents are not relevant.", "yellow"))
            else:
                 traceback.print_exc() # Show full trace for unexpected errors
            rag_context = "" # Discard context on error


    # === 4. Web Search / Deep Research ===
    # Check for real-time need (Using BooleanOutputParser)
    needs_realtime = True # Default assumption
    try:
        print(colored("Checking for real-time data need...", "cyan"))
        realtime_check_llm = main_llm_langchain
        # Updated prompt for BooleanOutputParser
        realtime_check_prompt = PromptTemplate(
            template="""Does the question below strongly imply a need for CURRENT, up-to-the-minute information like stock prices, breaking news, or live market status? Answer ONLY with 'YES' or 'NO'.\n\nQuestion: {question}""",
            input_variables=["question"]
        )
        realtime_check_parser = BooleanOutputParser()
        realtime_check_chain = realtime_check_prompt | realtime_check_llm | realtime_check_parser

        # BooleanOutputParser returns True/False
        needs_realtime = realtime_check_chain.invoke({"question": query})

        print(colored(f"Needs real-time data check result: {'Yes' if needs_realtime else 'No'}", "cyan"))
    except Exception as e:
        print(colored(f"Error checking for real-time need: {e}", "red"))
        if "Invalid" in str(e) or "OutputParserException" in str(e):
            print(colored("Failed to parse real-time need. Assuming real-time IS needed.", "yellow"))
        else:
            traceback.print_exc()
        needs_realtime = True # Default to True on error

    # Decide whether to perform web step
    perform_web_step = True
    if grade == 1 and not needs_realtime:
        print(colored("Relevant RAG found and no immediate real-time data need identified. Proceeding with web search for verification/augmentation.", "green"))
        # perform_web_step = False # Uncomment to skip web step in this case

    if perform_web_step:
        if deep_search:
            # --- Deep Research Path ---
            print(colored("Initiating Deep Research...", 'magenta'))
            if stream_callback: stream_callback("Initiating Deep Research...\n")
            try:
                # Pass the stream_callback to the research method
                # Ensure 'researcher' uses Agno-compatible models internally if needed
                research_result = researcher.research(query, stream_callback=stream_callback)
                web_research_context = research_result.get("answer", "Deep research failed to produce a synthesized answer.")
                research_debug_log = research_result.get("debug_log", "")
                print(colored("Deep Research completed.", "green"))
                if stream_callback: stream_callback("Deep Research completed.\n")

            except Exception as e:
                error_msg = f"Critical Error during Deep Research execution: {e}"
                print(colored(error_msg, "red"))
                traceback.print_exc()
                web_research_context = f"Deep research encountered a critical error: {str(e)}"
                research_debug_log = f"{research_debug_log}\n\n--- CRITICAL ERROR ---\n{error_msg}\n{traceback.format_exc()}"
                if stream_callback:
                    stream_callback(f"--- DEEP RESEARCH CRITICAL ERROR: {e} ---\n")
        else:
            # --- Standard Web Search Path ---
            print(colored("Initiating Standard Web Search using Tavily/YFinance...", 'magenta'))
            # Use Agno compatible LLM for Agno Agent
            web_search_agent = Agent(
                model=tool_llm_agno,
                description="""You are a Financial Assistant specialized in retrieving real-time and web-based information using Tavily Search for general info/news and YFinance for specific stock data. Execute tool calls as needed. Synthesize the results factually. Current time: {current_datetime}""",
                markdown=True,
                search_knowledge=False,
                tools=[tavily_client, yf_tool],
                show_tool_calls=True,
                add_datetime_to_instructions=True,
            )
            try:
                history = memory.load_memory_variables({})["chat_history"]
                response = web_search_agent.run(query, chat_history=history)
                web_research_context = response.content
                print(colored(f"Web Search Agent Response: {web_research_context}", "magenta"))
                print(colored("Standard Web Search completed.", "green"))

            except Exception as e:
                error_msg = f"Error during Standard Web Search Agent execution: {str(e)}"
                print(colored(error_msg, "red"))
                traceback.print_exc()
                web_research_context = f"Standard web search encountered an error: {str(e)}"

    # === 5. Synthesis ===
    print(colored("Synthesizing final answer...", "cyan"))
    # Use Agno compatible LLM for Agno Agent
    synthesis_agent = Agent(
        model=main_llm_agno,
        description="""You are a Financial Analyst Synthesizer. Combine information from internal knowledge (RAG Context) and web research (Web/Deep Research Context) to answer the user's original query comprehensively. Prioritize accuracy and recent information. Format clearly using Markdown.""",
        markdown=True,
    )

    try:
        synthesis_prompt_input = f"""Original Query: {query}

        --- Information from Knowledge Base (RAG Context) ---
        {rag_context if rag_context else "No relevant information found in internal documents."}

        --- Information from Web/Deep Research Context ---
        {web_research_context if web_research_context else "No information gathered from web search or deep research."}

        ---

        Synthesize the above information to answer the original query comprehensively and accurately. Structure the response clearly using Markdown. If conflicting information exists, highlight it or prioritize the most recent/reliable source (often the web context for current data). Respond directly to the user.
        """
        print(colored(f"SYNTHESIS PROMPT INPUT LENGTH: {len(synthesis_prompt_input)} chars", "grey"))

        history = memory.load_memory_variables({})["chat_history"]
        final_response = synthesis_agent.run(synthesis_prompt_input, chat_history=history)
        final_answer = final_response.content

    except Exception as e:
        print(colored(f"Error during final synthesis: {e}", "red"))
        traceback.print_exc()
        final_answer = f"Sorry, I encountered an error while synthesizing the final answer: {str(e)}"

    print(colored("Processing complete.", "white", attrs=["bold"]))

    return {
        "answer": final_answer,
        "deep_research_log": research_debug_log
        }

# Remove the entire testing block below
# Remove the entire testing block below
# Example of how to potentially run this file directly (for testing)
# if __name__ == "__main__":
#     print("Testing process_query_flow...")
#     # test_query = "Is Nio stock a good investment right now?"
#     # test_query = "What is the weather in London?" # Test small talk / tool use
#     test_query = "give me the stock list with date to invest and tell me the year" # Test grading failure

#     # Create a dummy memory for testing
#     test_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

#     print("\n--- Testing Standard Search ---")
#     result_standard = process_query_flow(test_query, test_memory, deep_search=False)
#     print("\nStandard Search Final Answer:")
#     console.print(Markdown(result_standard["answer"]))
#     print("-" * 30)
#
#     print("\n--- Testing Deep Research (will log to console) ---")
#     result_deep = process_query_flow(test_query, test_memory, deep_search=True)
#     print("\nDeep Research Final Answer:")
#     console.print(Markdown(result_deep["answer"]))
#     print("\nDeep Research Debug Log (excerpt):")
#     print(result_deep.get("deep_research_log", "No log returned")[:1000] + "...") # Safely get log
#     print("-" * 30)


# --- Pydantic Model for Request Body ---
class QueryRequest(BaseModel):
    query: str
    deep_search: bool = False # Default to False if not provided

# --- FastAPI App Setup ---
app = FastAPI(
    title="Financial Assistant API",
    description="API endpoint for the AI Financial Assistant",
    version="1.0.0"
)

# --- CORS Configuration ---
# Keep CORS for development or if API is accessed from other origins
origins = [
    "http://localhost:3000", # React dev server
    "http://localhost:8000", # Allow requests from the app itself if needed
    # Add deployed frontend URL if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory storage for conversation memory ---
# For production, consider a more robust session management solution
conversation_memory_store = {}

def get_or_create_memory(session_id: str = "default_session") -> ConversationBufferMemory:
    """Gets or creates a memory buffer for a session."""
    if session_id not in conversation_memory_store:
        conversation_memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True # Important for Langchain chains expecting message objects
        )
    return conversation_memory_store[session_id]

# --- API Endpoint ---
@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Handles user queries, processes them through the agent flow,
    and returns the AI's response.
    """
    try:
        print(f"Received query: {request.query}, Deep Search: {request.deep_search}")
        memory = get_or_create_memory()
        answer = process_query_flow(
            query=request.query,
            memory=memory,
            deep_search=request.deep_search
        )

        print(f"Generated answer: {answer}")
        return {"answer": answer}

    except Exception as e:
        print(colored(f"API Error: {str(e)}", "red"))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- Serve Static Frontend Files (Add this section) ---
# Define the path to the React build directory relative to run.py
# Adjust the path separators and levels ('..') as necessary based on your structure
frontend_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Frontend_React', 'build'))

# Check if the build directory exists before mounting
if os.path.exists(frontend_build_dir):
    print(colored(f"Serving static files from: {frontend_build_dir}", "cyan"))
    # Mount the static files directory (serving CSS, JS, images, etc.)
    # The path "/static" here means files in the build/static folder will be available at http://localhost:8000/static/...
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_build_dir, "static")), name="static")

    # Catch-all route to serve index.html for any other GET request
    # This is crucial for client-side routing (React Router)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        index_path = os.path.join(frontend_build_dir, 'index.html')
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            # Handle case where index.html is not found (optional)
            raise HTTPException(status_code=404, detail="Frontend index.html not found")
else:
    print(colored(f"Warning: Frontend build directory not found at {frontend_build_dir}. Static file serving disabled.", "yellow"))
    print(colored("Run 'npm run build' in the Frontend_React directory.", "yellow"))


# ... (rest of your existing functions like process_query_flow, display_tool_calls, etc.) ...

# --- Main Execution Block (for running with uvicorn) ---
# Keep this block if you want to run the server using 'python run.py'
if __name__ == "__main__":
    import uvicorn
    print(colored("Starting FastAPI server...", "cyan"))
    # Ensure the app object used here is the FastAPI instance 'app'
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True) # Use reload for development
    import uvicorn
    print(colored("Starting FastAPI server...", "cyan"))
    # Ensure the app object used here is the FastAPI instance 'app'
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True) # Use reload for development