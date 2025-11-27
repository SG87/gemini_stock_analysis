# Gemini Stock Analysis

A Python project that uses Google's Gemini API, Model Context Protocol (MCP), and vector databases to analyze stock data from Google Sheets.

## Features

- **Google Sheets Integration**: Reads stock data from Google Sheets
- **Gemini API**: Uses Google's Gemini AI for analysis and insights
- **MCP Integration**: Implements Model Context Protocol for structured AI interactions
- **Vector Database**: Stores and retrieves embeddings using ChromaDB for semantic search

## Prerequisites

- **Python 3.12** (required - this project specifically uses Python 3.12)
  - Check your version: `python3.12 --version`
  - See [SETUP.md](SETUP.md) for installation instructions
- Poetry (for dependency management)
- Google Cloud credentials (for Google Sheets API)
- Gemini API key

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Configure Poetry to use Python 3.12:
```bash
poetry env use python3.12
# Verify: poetry run python --version
```

3. Install dependencies:
```bash
poetry install
```

4. Set up environment variables:

Have a look look at SETUP.md to configure the below variables to input in `.env`

## Configuration

Create a `.env` file with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_SHEETS_ID=your_google_sheet_id_here
GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
CHROMA_DB_PATH=./chroma_db
```

## Usage

Before running, verify Python 3.12 is being used:
```bash
poetry run python check_python.py
```

Then run the application:
```bash
poetry run gemini-stock-analysis
```

Or activate the virtual environment:
```bash
poetry shell
python -m gemini_stock_analysis.main
```

## Project Structure

```
src/
├── __init__.py
├── main.py                 # Main application entry point
├── config.py               # Configuration management
├── sheets/
│   ├── __init__.py
│   └── reader.py          # Google Sheets integration
├── gemini/
│   ├── __init__.py
│   └── client.py          # Gemini API client
├── mcp/
│   ├── __init__.py
│   └── handler.py         # MCP protocol handler
└── vector_db/
    ├── __init__.py
    └── store.py           # Vector database operations
```

## Development

Run tests:
```bash
poetry run pytest
```

Format code:
```bash
poetry run black .
poetry run ruff check --fix .
```

## License

MIT

