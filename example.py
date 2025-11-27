"""Example script demonstrating how to use the components individually."""

from gemini_stock_analysis.config import get_settings
from gemini_stock_analysis.gemini.client import GeminiClient
from gemini_stock_analysis.mcp.handler import MCPHandler
from gemini_stock_analysis.sheets.reader import SheetsReader
from gemini_stock_analysis.vector_db.store import VectorStore


def example_usage():
    """Example of using each component."""
    # Load settings
    settings = get_settings()

    # Example 1: Read from Google Sheets
    print("Example 1: Reading from Google Sheets")
    sheets_reader = SheetsReader(settings)
    df = sheets_reader.read_sheet()
    print(f"Loaded {len(df)} rows")
    print(df.head())

    # Example 2: Use Gemini for analysis
    print("\nExample 2: Using Gemini API")
    gemini_client = GeminiClient(settings)
    response = gemini_client.analyze("What is artificial intelligence?")
    print(response)

    # Example 3: Use MCP handler
    print("\nExample 3: Using MCP Handler")
    mcp_handler = MCPHandler(settings)
    mcp_handler.add_context("system", "You are a helpful assistant.")
    mcp_handler.add_context("user", "Hello!")
    context = mcp_handler.get_context_string()
    print(context)

    # Example 4: Use Vector Store
    print("\nExample 4: Using Vector Store")
    vector_store = VectorStore(settings, collection_name="example")
    vector_store.add_documents(
        documents=["Apple stock is up 5%", "Microsoft reported strong earnings"],
        ids=["doc1", "doc2"],
    )
    results = vector_store.search("technology stocks", n_results=2)
    print(results)


if __name__ == "__main__":
    example_usage()


