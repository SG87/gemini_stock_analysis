"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini API
    gemini_api_key: str

    # Google Sheets
    google_sheets_id: str
    google_credentials_path: str = "./credentials.json"

    # Vector Database
    chroma_db_path: str = "./chroma_db"

    # MCP Configuration
    mcp_server_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        """Initialize settings and validate paths."""
        super().__init__(**kwargs)
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        creds_path = Path(self.google_credentials_path)
        if not creds_path.exists():
            raise FileNotFoundError(
                f"Google credentials file not found at {self.google_credentials_path}. "
                "Please download your credentials.json from Google Cloud Console."
            )

        # Create chroma_db directory if it doesn't exist
        db_path = Path(self.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings instance."""
    try:
        return Settings()
    except Exception as e:
        error_msg = str(e)
        if "Field required" in error_msg or "validation error" in error_msg.lower():
            print("\n" + "=" * 80)
            print("CONFIGURATION ERROR: Missing required environment variables")
            print("=" * 80)
            print("\nPlease create a .env file in the project root with the following variables:")
            print("\n  GEMINI_API_KEY=your_gemini_api_key_here")
            print("  GOOGLE_SHEETS_ID=your_google_sheet_id_here")
            print("  GOOGLE_CREDENTIALS_PATH=./credentials.json")
            print("\nSee SETUP.md for detailed instructions on how to obtain these values.")
            print("=" * 80 + "\n")
        raise
