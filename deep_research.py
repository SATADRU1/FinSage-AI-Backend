# deep_research.py
import time
from tavily import TavilyClient
from agno.agent import Agent
from vars import get_llm_id, get_llm_provider, MAX_DEPTH, MAX_SEARCH_CALLS, NUM_SUBQUESTIONS
import os
from typing import List, Dict, Any, Optional, Callable # Import Callable
from dotenv import load_dotenv
from termcolor import colored
from rich.console import Console
from rich.markdown import Markdown
import re
import json
from datetime import datetime
from agno.tools.yfinance import YFinanceTools

load_dotenv()
console = Console()

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


class DeepResearch:
    def __init__(self, max_depth=MAX_DEPTH, max_search_calls=MAX_SEARCH_CALLS):
        self.reasoning_model = get_llm_provider(get_llm_id("reasoning"))
        self.analysis_model = get_llm_provider(get_llm_id("remote"))
        self.max_depth = max_depth
        self.max_search_calls = max_search_calls
        self.search_calls_made = 0
        self.debug_log = []
        self.user_prompt = ""

    # Modified _log method
    def _log(self, message, color=None, attrs=None, stream_callback: Optional[Callable[[str], None]] = None):
        """Log a message to console, debug log, and optionally stream via callback."""
        log_entry = message # Store the raw message
        if color:
            colored_msg = colored(message, color, attrs=attrs)
            print(colored_msg)
        else:
            print(message)

        self.debug_log.append(log_entry) # Append raw message to internal log

        # If a callback is provided, call it with the raw message
        if stream_callback:
            try:
                # Add newline for better streaming display formatting
                stream_callback(log_entry + "\n")
            except Exception as e:
                # Avoid crashing backend if streaming fails, just print error
                print(colored(f"--- STREAMING CALLBACK ERROR: {e} ---", "red"))

    def _parse_subquestions(self, response_content: str, num_questions: int) -> List[str]:
        """Robustly parse numbered list of subquestions from LLM response."""
        # (Keep existing implementation - no changes needed here)
        subquestions = []
        lines = response_content.strip().split('\n')
        for line in lines:
            match = re.match(r"^\s*\d+[.)]?\s*(.*)", line)
            if match:
                question = match.group(1).strip().strip('"\'')
                if question:
                    subquestions.append(question)
        if not subquestions:
            for line in lines:
                cleaned = line.strip().lstrip("- ").lstrip("* ").strip('"\'')
                if cleaned:
                    subquestions.append(cleaned)
        return subquestions[:num_questions]


    # Modified _generate_subquestions
    def _generate_subquestions(self, query: str, num_questions=NUM_SUBQUESTIONS, stream_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """Break down query into subquestions, using the stream callback for logging."""
        agent = Agent(
            model=self.analysis_model,
            description="You are an expert research planner...",
            # system_message=SYSTEM_PROMPT_FOR_SUBQUESTIONS
        )
        self.user_prompt = query # Store user prompt for context in analysis

        prompt = f"""
        You need to research the following complex topic: "{query}"
        Break this down into exactly {num_questions} specific...
        Return ONLY the numbered list... 
        
        CONTEXT : 
        
        Today's date is {datetime.now().strftime("%Y-%m-%d")}
        Always search for the latest information, considering the current date.
        """ # (Keep existing prompt)

        try:
            self._log(f"Generating {num_questions} subquestions for: '{query}'", "cyan", stream_callback=stream_callback)
            response = agent.run(prompt)
            content = response.content
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()

            subquestions = self._parse_subquestions(content, num_questions)

            self._log(f"Generated {len(subquestions)} subquestions:", "cyan", stream_callback=stream_callback)
            for idx, q in enumerate(subquestions):
                self._log(f"  {idx+1}. {q}", "yellow", stream_callback=stream_callback)
            return subquestions

        except Exception as e:
            self._log(f"Error generating subquestions: {e}", "red", stream_callback=stream_callback)
            return []

    # Modified _should_decompose (added stream_callback, though not directly used for logging here)
    def _should_decompose(self, subquestion: str, context: str, stream_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Decide if a subquestion needs further decomposition."""
        agent = Agent(model=self.analysis_model)
        prompt = f"""
        Consider the subquestion: "{subquestion}"
        Initial research provided this context:
        ---
        {context[:1500]}...
        ---
        Based *only* on the subquestion and the initial context, is the subquestion still too broad...?
        Answer with only YES or NO.
        """
        try:
            response = agent.run(prompt)
            return "YES" in response.content.upper()
        except Exception as e:
            # Log the error using the callback
            self._log(f"Error checking decomposition for '{subquestion}': {e}", "red", stream_callback=stream_callback)
            return False

    # Modified _research_subquestion
    def _research_subquestion(self, subquestion: str, depth=0, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Research subquestion, using stream callback for logging."""
        self._log(
            f"\n{'  ' * depth}Researching (Depth {depth}): {subquestion}", "green", stream_callback=stream_callback)

        if self.search_calls_made >= self.max_search_calls:
            self._log(
                f"{'  ' * depth}Skipping research: Max search calls ({self.max_search_calls}) reached.", "red", stream_callback=stream_callback)
            return {
                "subquestion": subquestion, "summary": "Max search calls reached.",
                "search_results": None, "context": "", "additional_info": {}
            }

        context = ""
        search_results = None
        tool_outputs = {}

        # --- Tool Integration ---
        agent = Agent(model=self.reasoning_model)
        relevance_prompt = f"""
        Analyze this question: "{subquestion}"
        Is this question related to stock prices... using YFinance...?
        Answer with only YES or NO.
        """
        try:
            relevance_response = agent.run(relevance_prompt)
            is_yfinance_relevant = "YES" in relevance_response.content.upper()

            if is_yfinance_relevant:
                self._log(f"{'  ' * depth}YFinance determined to be relevant for: {subquestion}", "blue", stream_callback=stream_callback)
                yf_agent = Agent(model=self.reasoning_model, tools=[yf_tool], show_tool_calls=True, markdown=True)
                try:
                    self._log(f"{'  ' * depth}Calling YFinance for: {subquestion}", "blue", stream_callback=stream_callback)
                    yf_response = yf_agent.run(subquestion)
                    yf_output = yf_response.content
                    if "404 Client Error:" in yf_output: # Check for common yfinance error
                         self._log(f"{'  ' * depth}YFinance returned 404 error. Falling back.", "red", stream_callback=stream_callback)
                         raise ValueError("YFinance 404 error") # Raise to trigger fallback

                    tool_outputs["yfinance_data"] = yf_output
                    context += f"\nYFinance Tool Output:\n{yf_output}\n"
                    self._log(f"{'  ' * depth}YFinance Output received.", "magenta", stream_callback=stream_callback)
                    # Note: show_tool_calls=True in Agno might print, but we log receipt here.
                    self.search_calls_made += 1 # Increment after successful call
                except Exception as e:
                    self._log(f"{'  ' * depth}YFinance agent failed: {e}", "red", stream_callback=stream_callback)
                    # Decide if fallback to Tavily is needed here or handled below
                    is_yfinance_relevant = False # Treat as not relevant if failed
            else:
                self._log(f"{'  ' * depth}YFinance determined not relevant.", "yellow", stream_callback=stream_callback)

        except Exception as e:
            self._log(f"{'  ' * depth}Error determining YFinance relevance: {e}", "red", stream_callback=stream_callback)
            is_yfinance_relevant = False # Assume not relevant on error

        # --- Tavily Web Search (Run if YFinance not relevant, failed, or general search needed) ---
        # Simplified logic: Always run Tavily unless YFinance provided a definitive answer (hard to judge, so usually run)
        if self.search_calls_made < self.max_search_calls:
            self._log(f"{'  ' * depth}Performing Tavily search for: {subquestion}", "blue", stream_callback=stream_callback)
            try:
                search_results = tavily_client.search(query=subquestion, search_depth="advanced", max_results=5)
                self.search_calls_made += 1
                if search_results and search_results.get("results"):
                    context += "\nWeb Search Results (Tavily):\n" + "\n\n".join([f"Source: {r.get('url', 'N/A')}\nContent: {r.get('content', '')}" for r in search_results["results"]])
                    self._log(f"{'  ' * depth}Tavily search successful.", "magenta", stream_callback=stream_callback)
                else:
                    self._log(f"{'  ' * depth}Tavily search returned no results.", "yellow", stream_callback=stream_callback)
            except Exception as e:
                self._log(f"{'  ' * depth}Tavily search failed: {e}", "red", stream_callback=stream_callback)
                context += "\nWeb search failed."
        else:
             self._log(f"{'  ' * depth}Skipping Tavily search: Max search calls reached.", "red", stream_callback=stream_callback)


        # --- Recursive Decomposition ---
        additional_info = {}
        if depth < self.max_depth and self.search_calls_made < self.max_search_calls:
            # Pass callback to _should_decompose
            if self._should_decompose(subquestion, context, stream_callback=stream_callback):
                self._log(f"{'  ' * depth}Further decomposing: {subquestion}", "magenta", stream_callback=stream_callback)
                 # Pass callback to _generate_subquestions
                sub_subquestions = self._generate_subquestions(subquestion, num_questions=2, stream_callback=stream_callback)

                sub_findings = {}
                for sub_sq in sub_subquestions:
                    # Recursive call - pass callback
                    sub_result = self._research_subquestion(sub_sq, depth=depth + 1, stream_callback=stream_callback)
                    sub_findings[sub_sq] = sub_result
                additional_info["sub_research"] = sub_findings
            else:
                self._log(f"{'  ' * depth}Decomposition not needed for: {subquestion}", "yellow", stream_callback=stream_callback)

        # --- Analysis/Summarization ---
        # Pass callback to _analyze_findings
        summary = self._analyze_findings(subquestion, context, additional_info, stream_callback=stream_callback)

        return {
            "subquestion": subquestion,
            "search_results": search_results,
            "tool_outputs": tool_outputs,
            "context": context,
            "summary": summary,
            "additional_info": additional_info
        }

    # Modified _analyze_findings
    def _analyze_findings(self, subquestion: str, context: str, additional_info: Dict, stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """Analyze findings, using stream callback for logging."""
        self._log(f"Analyzing findings for: {subquestion}", "cyan", stream_callback=stream_callback)
        agent = Agent(
            model=self.reasoning_model, # Use reasoning model for analysis correctness
            description="You are a research analyst...",
        )

        sub_research_summary = ""
        if "sub_research" in additional_info:
            sub_research_summary = "\n\n--- Findings from Deeper Analysis ---\n"
            for sub_sq, sub_result in additional_info["sub_research"].items():
                sub_research_summary += f"\nRegarding '{sub_sq}':\n{sub_result.get('summary', 'No summary available.')}\n"

        prompt = f"""
        Please analyse the correctness of the information provided and synthesize the following information to answer the specific subquestion: "{subquestion}"

        --- Information Gathered ---
        {context[:8000]}...
        {sub_research_summary}
        ---

        Instructions:
        1. Focus *only* on answering the subquestion: "{subquestion}"
        2. Analyse the correctness of the information provided. If the information is incorrect with respect to the subquestion and the user prompt (which is "{self.user_prompt}") in general... provide the correct information.
        3. Create a clear, concise, and factual summary...
        """ # (Keep existing prompt structure)
        try:
            response = agent.run(prompt)
            self._log(f"Analysis complete for: {subquestion}", "green", stream_callback=stream_callback)
            return response.content
        except Exception as e:
            self._log(f"Error analyzing findings for '{subquestion}': {e}", "red", stream_callback=stream_callback)
            return f"Error summarizing findings for '{subquestion}'."

    # Modified _synthesize_research
    def _synthesize_research(self, main_query: str, research_results: Dict[str, Dict], stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """Synthesize final answer, using stream callback for logging."""
        self._log("Synthesizing final research report...", "cyan", stream_callback=stream_callback)
        agent = Agent(
            model=self.analysis_model,
            description="You are a senior research analyst...",
        )

        findings_context = ""
        for subq, result in research_results.items():
            findings_context += f"\n\n## Research Findings for: {subq}\n\n{result.get('summary', 'No summary available.')}"
            # Optional: Log tool outputs if needed during synthesis debug
            # if result.get('tool_outputs'):
            #     self._log(f"Tool outputs for {subq}: {json.dumps(result['tool_outputs'], indent=2)}", "grey", stream_callback=stream_callback)


        prompt = f"""
        You have been tasked with researching the question: "{main_query}"
        The research was broken down... summaries were generated for each:
        --- Combined Research Summaries ---
        {findings_context}
        ---
        Instructions:
        1. Synthesize *all* the provided summaries... main question: "{main_query}".
        ... (Keep existing prompt) ...
        9. Format the output using Markdown for readability.
        """
        try:
            response = agent.run(prompt)
            self._log("Final synthesis complete.", "green", stream_callback=stream_callback)
            return response.content
        except Exception as e:
            self._log(f"Error synthesizing final answer: {e}", "red", stream_callback=stream_callback)
            return f"Error synthesizing the final research report: {e}"

    # Modified research method signature
    def research(self, query: str, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Execute deep research, passing stream callback down."""
        self.debug_log = [] # Clear log for new research
        self.search_calls_made = 0 # Reset counter
        # Pass callback to initial log
        self._log(f"\n=== Starting Deep Research on: {query} ===", "blue", attrs=["bold"], stream_callback=stream_callback)

        # Step 1: Pass callback to generate subquestions
        subquestions = self._generate_subquestions(query, stream_callback=stream_callback)
        if not subquestions:
            self._log("Failed to generate subquestions. Aborting deep research.", "red", stream_callback=stream_callback)
            # Ensure final dict structure is consistent
            return {
                "query": query,
                "answer": "Could not perform deep research due to an issue generating subquestions.",
                "debug_log": "\n".join(self.debug_log),
                 "subquestions": [],
                 "subquestion_results": {}
            }

        # Step 2: Pass callback to research each subquestion
        subquestion_results = {}
        for sq in subquestions:
            if self.search_calls_made >= self.max_search_calls:
                self._log(f"Max search calls ({self.max_search_calls}) reached. Skipping remaining subquestions.", "red", stream_callback=stream_callback)
                subquestion_results[sq] = {"summary": "Skipped due to max search call limit."}
                continue
            # Pass callback here
            subquestion_results[sq] = self._research_subquestion(sq, depth=0, stream_callback=stream_callback)

        # Step 3: Pass callback to synthesize findings
        final_answer = self._synthesize_research(query, subquestion_results, stream_callback=stream_callback)

        self._log("\n=== Deep Research Complete ===", "blue", attrs=["bold"], stream_callback=stream_callback)

        total_calls = self.search_calls_made
        self._log(f"Total search calls made: {total_calls}", "cyan", stream_callback=stream_callback)

        # Return the full results including the internally collected debug_log
        return {
            "query": query,
            "subquestions": subquestions,
            "subquestion_results": subquestion_results,
            "answer": final_answer,
            "debug_log": "\n".join(self.debug_log) # Still return the complete log
        }


# Keep the instance creation and __main__ block as is
deep_research = DeepResearch()

if __name__ == "__main__":
    # Example usage (will print to console, no Streamlit callback here)
    # query = "Compare the financial health and future prospects of Tesla and Nio."
    with (open('./docs/query.md', 'r')) as f:
        query = f.readline().strip() # Ensure no trailing newline from file

    start_time = time.time()
    # In direct execution, stream_callback is None
    result = deep_research.research(query, stream_callback=None)
    end_time = time.time()

    # Use internal _log for consistency if needed, or just print
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    console.print(Markdown(colored("\n=== Final Research Answer ===\n", "green", attrs=["bold"])))
    console.print(Markdown(result["answer"]))

    debug_log_path = './docs/debug_log.txt'
    with open(debug_log_path, 'w') as f:
        f.write(result["debug_log"])
    print(f"Debug log saved to {debug_log_path}")

    output_path = './docs/outputs.md'
    with open(output_path, 'a') as f:
        f.write("\n\n---\n\n**Query:** " + query + "\n\n" + result["answer"])
    print(f"Output appended to {output_path}")