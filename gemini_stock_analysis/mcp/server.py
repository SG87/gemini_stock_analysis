import uvicorn

from fastapi import FastAPI, Depends, Security, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_mcp import FastApiMCP

# Define auth
security = HTTPBearer()


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


# TODO: Create proper tools that connect to the Chromadb
@app.get("/list_name", operation_id="list_names")
async def list_names():
    """
    List names of analysts.
    """
    return ["Stijn", "Pieter", "Johan"]


if __name__ == "__main__":
    # 2. MCP setup
    mcp = FastApiMCP(app, name="GeminiStockAnalysis", include_operations=[list_names])
    mcp.mount_sse()

    uvicorn.run(app, host="0.0.0.0", port=8000)
