"""Google Sheets reader for fetching stock data."""

import os
from typing import Any, Dict, List

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..config import Settings

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


class SheetsReader:
    """Reads data from Google Sheets."""

    def __init__(self, settings: Settings):
        """Initialize the Sheets reader with settings."""
        self.settings = settings
        self.service = self._get_service()

    def _get_service(self) -> Any:
        """Get authenticated Google Sheets service."""
        creds = None
        # The file token.json stores the user's access and refresh tokens.
        token_path = "token.json"
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.settings.google_credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(token_path, "w") as token:
                token.write(creds.to_json())

        return build("sheets", "v4", credentials=creds)

    def read_sheet(self, range_name: str = "A1:Z1000") -> pd.DataFrame:
        """
        Read data from the configured Google Sheet.

        Args:
            range_name: The A1 notation range to read (e.g., "Sheet1!A1:Z1000")

        Returns:
            DataFrame containing the sheet data
        """
        try:
            sheet = self.service.spreadsheets()
            result = (
                sheet.values()
                .get(spreadsheetId=self.settings.google_sheets_id, range=range_name)
                .execute()
            )
            values = result.get("values", [])

            if not values:
                return pd.DataFrame()

            # First row as headers
            headers = values[0]
            data = values[1:] if len(values) > 1 else []

            # Pad rows to match header length
            max_cols = len(headers)
            padded_data = [row + [""] * (max_cols - len(row)) for row in data]

            df = pd.DataFrame(padded_data, columns=headers)
            return df

        except HttpError as error:
            print(f"An error occurred: {error}")
            raise

    def get_all_sheets(self) -> List[Dict[str, str]]:
        """
        Get metadata about all sheets in the spreadsheet.

        Returns:
            List of dictionaries with sheet metadata
        """
        try:
            sheet_metadata = (
                self.service.spreadsheets()
                .get(spreadsheetId=self.settings.google_sheets_id)
                .execute()
            )
            sheets = sheet_metadata.get("sheets", [])
            return [
                {
                    "title": sheet["properties"]["title"],
                    "sheet_id": sheet["properties"]["sheetId"],
                }
                for sheet in sheets
            ]
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise


