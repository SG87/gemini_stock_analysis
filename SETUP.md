# Setup Guide

This guide will help you set up the Gemini Stock Analysis project.

## Prerequisites

1. **Python 3.12**: This project requires Python 3.12 specifically
   ```bash
   python3.12 --version
   ```
   
   If Python 3.12 is not installed, you can install it using one of these methods:
   
   **Option A: Using Homebrew (macOS)**
   ```bash
   brew install python@3.12
   ```
   
   **Option B: Using pyenv (recommended for managing multiple Python versions)**
   ```bash
   # Update pyenv first
   brew upgrade pyenv
   # Install Python 3.12
   pyenv install 3.12.9
   # Set it for this project
   cd /path/to/gemini_stock_analysis
   pyenv local 3.12.9
   ```
   
   **Option C: Download from python.org**
   - Visit https://www.python.org/downloads/
   - Download Python 3.12.x for your operating system
   - Follow the installation instructions

2. **Poetry**: Install Poetry for dependency management
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

## Step 1: Configure Poetry to Use Python 3.12

Make sure Poetry uses Python 3.12 for this project:

```bash
# If you installed Python 3.12 via Homebrew
poetry env use $(brew --prefix python@3.12)/bin/python3.12

# If you installed Python 3.12 via pyenv
poetry env use $(pyenv which python3.12)

# Or specify the full path to your Python 3.12 installation
poetry env use /path/to/python3.12
```

Verify the Python version:
```bash
poetry run python --version
# Should output: Python 3.12.x
```

## Step 2: Install Dependencies

```bash
poetry install
```

This will create a virtual environment with Python 3.12 and install all required packages.

## Step 3: Google Cloud Setup

### Get Google Sheets API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Sheets API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click "Enable"
4. Create credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop app" as the application type
   - Download the JSON file and save it as `credentials.json` in the project root

### Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

## Step 4: Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_SHEETS_ID=your_google_sheet_id_here
GOOGLE_CREDENTIALS_PATH=./credentials.json
CHROMA_DB_PATH=./chroma_db
```

### Getting Your Google Sheet ID

1. Open your Google Sheet
2. Look at the URL: `https://docs.google.com/spreadsheets/d/SHEET_ID_HERE/edit`
3. Copy the `SHEET_ID_HERE` part

## Step 5: Prepare Your Google Sheet

Your Google Sheet should have:
- A header row with column names
- Data rows below the header
- Example columns: `Symbol`, `Price`, `Volume`, `Change`, etc.

## Step 6: Run the Application

```bash
poetry run gemini-stock-analysis
```

Or activate the virtual environment first:

```bash
poetry shell
python -m gemini_stock_analysis.main
```

## First Run

On the first run, the application will:
1. Open a browser window for Google OAuth authentication
2. Ask you to sign in and grant permissions
3. Save the token for future runs

## Troubleshooting

### "Credentials file not found"
- Make sure `credentials.json` is in the project root
- Check that `GOOGLE_CREDENTIALS_PATH` in `.env` points to the correct location

### "Permission denied" errors
- Make sure you've enabled Google Sheets API in Google Cloud Console
- Check that your OAuth credentials are set up correctly

### "API key invalid" errors
- Verify your Gemini API key is correct
- Make sure there are no extra spaces in the `.env` file

### Import errors
- Make sure you've run `poetry install`
- Activate the virtual environment with `poetry shell`


