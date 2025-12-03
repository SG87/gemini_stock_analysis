"""Main application entry point."""

import sys

import pandas as pd

from gemini_stock_analysis.config import get_settings
from gemini_stock_analysis.gemini.client import GeminiClient
from gemini_stock_analysis.sheets.reader import SheetsReader
from gemini_stock_analysis.vector_db.store import VectorStore


def main() -> None:
    """Main application function."""
    try:
        # Load settings
        settings = get_settings()
        print("✓ Configuration loaded")

        # Initialize components
        print("Initializing components...")
        sheets_reader = SheetsReader(settings)
        print("✓ Google Sheets reader initialized")

        gemini_client = GeminiClient(settings)
        print("✓ Gemini client initialized")

        vector_store = VectorStore(settings)
        print("✓ Vector database initialized")

        # Read data from Google Sheets
        print("\nReading data from Google Sheets...")
        df = sheets_reader.read_sheet()
        print(f"✓ Loaded {len(df)} rows from Google Sheets")

        if df.empty:
            print("Warning: No data found in Google Sheets")
            return

        # Display sample data
        print("\nSample data:")
        print(df.head().to_string())

        # Convert DataFrame rows to documents for vector storage
        print("\nProcessing data for vector storage...")
        documents = []
        metadatas = []

        for idx, row in df.iterrows():
            # Create a text representation of each row
            doc_text = f"Stock data row {idx}: " + " | ".join(
                [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            )
            documents.append(doc_text)

            # Store metadata
            metadata = {col: str(val) for col, val in row.items() if pd.notna(val) or val != ""}
            metadata["row_index"] = str(idx)
            metadatas.append(metadata)

        # Generate embeddings and store in vector database
        print("Generating embeddings...")
        embeddings = gemini_client.generate_embeddings(documents)

        print("Embeddings size:", len(embeddings[0]))

        # Store in vector database
        print("Storing in vector database...")
        # Always provide embeddings to avoid ChromaDB CoreML/ONNX issues on macOS
        if not embeddings:
            # Generate fallback embeddings if Gemini embeddings are not available
            import hashlib

            print("Using fallback embeddings (Gemini embeddings not available)")
            embeddings = []
            for doc in documents[:10]:
                hash_obj = hashlib.sha256(doc.encode())
                hash_bytes = hash_obj.digest()
                embedding = [float(b) / 255.0 for b in hash_bytes[:128]] + [0.0] * (
                    128 - len(hash_bytes[:128])
                )
                embeddings.append(embedding)

        vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"row_{i}" for i in range(len(documents))],
        )
        print(f"✓ Stored {len(documents)} documents in vector database")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nPlease ensure you have:")
        print("1. Created a .env file with required configuration")
        print("2. Downloaded credentials.json from Google Cloud Console")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
