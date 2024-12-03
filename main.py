from fastapi import FastAPI, HTTPException, Query
from enum import Enum
from duckduckgo_search import DDGS
from typing import Annotated


app = FastAPI(
   title="LLM Chat API",
)

class ModelEnum(str, Enum):
    gpt_4o_mini = 'gpt-4o-mini'
    meta_llama = 'meta-llama'
    mistralai = 'mixtral'
    claude_3 = 'claude-3'

MODEL_MAPPING = {
   'gpt-4o-mini': 'gpt-4o-mini',
   'meta-llama': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
   'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
   'claude-3': 'claude-3-haiku-20240307'
}

@app.post("/chat")
async def chat(
    query: str,
    model: Annotated[ModelEnum, Query()] = ModelEnum.gpt_4o_mini
):
    try:
        actual_model = MODEL_MAPPING[model.value]
        print(actual_model) 
        results = DDGS().chat(query, model=actual_model)
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", tags=["chat"])
async def get_models():
   return {"models": list(ModelEnum)}

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)

