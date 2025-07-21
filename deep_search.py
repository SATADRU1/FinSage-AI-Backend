import asyncio
from typing import List, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

class DeepSearchProcessor:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.search_tool = DuckDuckGoSearchRun()
    
    async def perform_deep_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a comprehensive deep search with step-by-step processing
        
        Args:
            query (str): The user's search query
        
        Returns:
            Dict containing search steps and final result
        """
        # Step 1: Initial Query Analysis
        query_analysis_prompt = ChatPromptTemplate.from_template(
            "Analyze the following search query and break it down into key search components: {query}"
        )
        query_analysis_chain = query_analysis_prompt | self.llm | StrOutputParser()
        query_components = await query_analysis_chain.ainvoke({"query": query})
        
        # Step 2: Perform Initial Search
        search_results = self.search_tool.run(query)
        
        # Step 3: Result Refinement
        refinement_prompt = ChatPromptTemplate.from_template(
            "Given the original query: {query}\n"
            "And these search results: {results}\n"
            "Provide a concise, informative summary that directly answers the query."
        )
        refinement_chain = refinement_prompt | self.llm | StrOutputParser()
        refined_result = await refinement_chain.ainvoke({
            "query": query,
            "results": search_results
        })
        
        # Compile search process
        return {
            "steps": [
                {
                    "name": "Query Analysis",
                    "description": query_components
                },
                {
                    "name": "Initial Search",
                    "results": search_results
                },
                {
                    "name": "Result Refinement",
                    "final_result": refined_result
                }
            ],
            "final_result": refined_result
        }

# Example usage
async def run_deep_search(query: str, llm):
    processor = DeepSearchProcessor(llm)
    return await processor.perform_deep_search(query)
