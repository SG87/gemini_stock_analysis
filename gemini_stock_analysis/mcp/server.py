import uvicorn

from fastapi import FastAPI, Depends, Security, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_mcp import FastApiMCP

from gemini_stock_analysis.config import get_settings
from gemini_stock_analysis.vector_db.store import VectorStore

# Define auth
security = HTTPBearer()

# Initialize settings and vector store
settings = get_settings()
vector_store = VectorStore(settings=settings)


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify the provided token.
    For demonstration, we check against a hardcoded token.
    In production, replace this with proper token validation logic.
    Args:
        credentials: HTTPAuthorizationCredentials object containing the token
    Raises:
        HTTPException: If the token is invalid
    """
    if credentials.credentials != "SUPER_SECRET_TOKEN":
        raise HTTPException(status_code=401, detail="Invalid Token")


# 1. Your existing main application
app = FastAPI(title="Gemini Stock Analysis API")


@app.get("/health", dependencies=[Depends(verify_token)])
async def health():
    return {"status": "ok"}


@app.get("/search", operation_id="search_vector_db")
async def search_vector_db(query: str, n_results: int = 5):
    """
    Search the vector database for similar documents.
    """
    return vector_store.search(query=query, n_results=n_results)


if __name__ == "__main__":
    # 2. MCP setup
    mcp = FastApiMCP(app, name="GeminiStockAnalysis", include_operations=[search_vector_db])
    mcp.mount_sse()

    uvicorn.run(app, host="0.0.0.0", port=8000)
