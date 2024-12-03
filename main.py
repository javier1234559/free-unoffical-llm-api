from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from duckduckgo_search import DDGS
from fastapi.responses import JSONResponse

class ModelEnum(str, Enum):
    gpt_4o_mini = 'gpt-4o-mini'
    meta_llama = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    mistralai = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    claude_3 = 'claude-3-haiku-20240307'

app = FastAPI(
    title="Chat API",
    description="API để thực hiện cuộc trò chuyện",
    version="1.0.0",
    openapi_tags=[
        {"name": "chat", "description": "Thực hiện cuộc trò chuyện"}
    ]
)

class ChatRequest(BaseModel):
    query: str
    model: ModelEnum

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Thực hiện cuộc trò chuyện với mô hình được chọn

    Args:
        request (ChatRequest): Dữ liệu từ người dùng bao gồm câu hỏi và mô hình

    Returns:
        JSONResponse: Kết quả của cuộc trò chuyện
    """
    try:
        results = DDGS().chat(request.query, model=request.model)
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")

