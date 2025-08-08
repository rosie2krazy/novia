"""💰 Finance Agent - Your AI Stock Market Analyst!

This advanced finance agent provides comprehensive stock market analysis and financial insights.
The agent can analyze stocks, provide recommendations, and maintain conversation context.

The agent can:
- Analyze stock prices, trends, and performance
- Provide analyst recommendations and company information
- Search for latest financial news and market updates
- Maintain conversation context and memory across sessions
- Track user preferences and investment interests
- Answer follow-up questions about stocks and market conditions

Example queries to try:
- "Analyze AAPL stock and provide a recommendation"
- "What are the latest analyst recommendations for TSLA?"
- "Get company information for MSFT"
- "What's the latest news about NVDA?"
- "Compare the performance of GOOGL vs META"
- "What are the key financial metrics for AMZN?"

The agent uses:
- YFinance for real-time stock data and analysis
- DuckDuckGo for latest financial news and market updates
- Conversation memory for contextual responses
- User-specific session tracking
"""

from typing import Optional
import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.models.openrouter import OpenRouter
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage
from agno.media import Image

load_dotenv()



def get_finance_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    """Get a Finance Agent with Memory and YFinance tools."""

    # Use Gemini model via OpenRouter
    model = OpenRouter(id="google/gemini-2.0-flash-001", api_key=os.getenv("OPENROUTER_API_KEY"))

    # Initialize memory system
    memory = Memory(
        model=model,
        db=SqliteMemoryDb(table_name="finance_agent_memories", db_file="finance_agent.db"),
    )

    # Create the Finance Agent
    return Agent(
        name="finance_agent",
        session_id=session_id,  # Track session ID for persistent conversations
        user_id=user_id,
        model=model,
        storage=SqliteStorage(
            table_name="finance_agent_sessions", db_file="agent.db"
        ),  # Persist session data
        description="You are a professional Finance Agent and your goal is to provide comprehensive stock market analysis and financial insights.",
        instructions=[
            "1. Stock Analysis:",
            "   - Use YFinance tools to get real-time stock data, prices, and company information",
            "   - Provide detailed analysis of stock performance, trends, and key metrics",
            "   - Include analyst recommendations and price targets when available",
            "   - Compare stocks when multiple symbols are mentioned",
            "2. Market Research:",
            "   - Use DuckDuckGo to search for latest financial news and market updates",
            "   - Focus on reputable financial sources and recent information",
            "   - Cross-reference information from multiple sources when possible",
            "3. Image Analysis:",
            "   - You can understand and analyze all images that are sent to you",
            "   - If an image contains charts, graphs, or financial data, analyze them thoroughly",
            "   - Extract stock symbols, prices, trends, or any financial information from images",
            "   - Provide insights based on visual data like stock charts, company logos, or financial documents",
            "4. Context Management:",
            "   - Use get_chat_history tool to maintain conversation continuity",
            "   - Reference previous stock analyses and user preferences",
            "   - Keep track of stocks the user has shown interest in",
            "5. Response Quality:",
            "   - Structure responses with clear sections (Price Analysis, Company Info, News, etc.)",
            "   - Include specific numbers, percentages, and key metrics",
            "   - Provide actionable insights and recommendations",
            "   - Use bullet points and tables for better readability",
            "6. User Interaction:",
            "   - Ask for clarification if stock symbols are ambiguous",
            "   - Suggest related stocks or market sectors when relevant",
            "   - Proactively suggest follow-up analyses or comparisons",
            "7. Risk Management:",
            "   - Always include appropriate disclaimers about investment risks",
            "   - Emphasize that recommendations are for informational purposes only",
            "   - Suggest consulting with financial advisors for investment decisions",
        ],
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                company_info=True,
                company_news=True
            ),
            DuckDuckGoTools()
        ],
        markdown=True,  # Format messages in markdown
        show_tool_calls=True,
        add_history_to_messages=True,  # Adds chat history to messages
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        read_tool_call_history=True,
        num_history_responses=3,
        enable_user_memories=True,
        memory=memory,
        enable_agentic_memory=True,
    )
