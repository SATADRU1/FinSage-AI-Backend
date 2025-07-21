from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import yfinance as yf
from termcolor import colored

from dotenv import load_dotenv
load_dotenv()

class StockPriceInput(BaseModel):
    """Input for the stock price tool."""
    ticker: str = Field(...,
                        description="The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")


class StockPriceTool(BaseTool):
    name: str = "stock_price"
    description: str = "Get the current stock price for a given ticker symbol"
    args_schema: Type[BaseModel] = StockPriceInput

    def _run(self, ticker: str) -> str:
        """Get the current stock price for a given ticker symbol."""
        try:
            print(colored(f"Fetching stock price for {ticker}...", "blue"))
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")

            if data.empty:
                return f"Could not find stock price data for ticker {ticker}"

            current_price = data['Close'].iloc[-1]
            open_price = data['Open'].iloc[0]
            high_price = data['High'].max()
            low_price = data['Low'].min()
            volume = data['Volume'].iloc[-1]

            result = {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "volume": volume
            }

            return f"""Stock price information for {ticker}:
Current Price: ${result['current_price']}
Open: ${result['open']}
High: ${result['high']}
Low: ${result['low']}
Volume: {result['volume']:,}
"""
        except Exception as e:
            return f"Error fetching stock price for {ticker}: {str(e)}"

    async def _arun(self, ticker: str) -> str:
        """Async implementation of the stock price tool."""
        # For simplicity, we're just calling the sync version
        return self._run(ticker)


# Example usage
if __name__ == "__main__":
    # Create an instance of the tool
    stock_price_tool = StockPriceTool()

    # Example 1: Get Apple's stock price
    apple_result = stock_price_tool.run("AAPL")
    print(apple_result)

    # Example 2: Get Microsoft's stock price
    microsoft_result = stock_price_tool.run("MSFT")
    print(microsoft_result)

    # Example 3: Using the tool with LangChain Agent
    from langchain.agents import AgentType, initialize_agent
    from langchain_groq import ChatGroq

    # Uncomment and use your preferred LLM
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    tools = [stock_price_tool]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    result = agent.run("what stocks are currently profitable? Give me a list of all those stocks as well as their current price.")
    print(result)
