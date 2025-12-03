from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# MCP Imports
from mcp import ClientSession
from mcp.client.sse import sse_client

from gemini_stock_analysis.config import get_settings
# Gemini Imports

from gemini_stock_analysis.gemini import GeminiClient

# --- Configuration ---
# URL of the *other* FastAPI server (the one running the tools)
MCP_SERVER_URL = "http://localhost:8000/sse"

# Global state to hold the MCP connection
mcp_session: ClientSession | None = None
mcp_exit_stack = None


# --- Helper: Convert MCP Tools to Gemini Tools ---

# --- Lifecycle Manager ---
# This connects to the MCP server when your FastAPI app starts
# and disconnects when it shuts down.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_session, mcp_exit_stack
    from contextlib import AsyncExitStack

    print(f"🔌 Connecting to MCP Server at {MCP_SERVER_URL}...")

    try:
        mcp_exit_stack = AsyncExitStack()
        # Connect to SSE
        transport = await mcp_exit_stack.enter_async_context(sse_client(MCP_SERVER_URL))
        # Create Session
        mcp_session = await mcp_exit_stack.enter_async_context(
            ClientSession(transport[0], transport[1])
        )
        await mcp_session.initialize()
        print("✅ Connected to MCP Server!")

        yield  # Application is running now

    finally:
        print("🔌 Disconnecting from MCP Server...")
        if mcp_exit_stack:
            await mcp_exit_stack.aclose()


app = FastAPI(lifespan=lifespan)

# 2. Prepare Gemini Client
settings = get_settings()
gemini_client = GeminiClient(settings)


# --- Request Models ---
class ChatRequest(BaseModel):
    message: str


# --- The Chat Endpoint ---
@app.post("/chat")
async def chat(request: ChatRequest):
    if not mcp_session:
        raise HTTPException(status_code=503, detail="MCP Server not connected")

    response = await gemini_client.chat(message=request.message, mcp_session=mcp_session)

    return response

if __name__ == "__main__":
    import uvicorn

    # Run this client on port 8001 so it doesn't conflict with the MCP server on 8000
    uvicorn.run(app, host="0.0.0.0", port=5000)
