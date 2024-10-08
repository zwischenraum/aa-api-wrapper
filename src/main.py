import json
import os
from typing import Callable, Dict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

load_dotenv()

ALEPH_ALPHA_API_BASE = os.environ.get("ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com")

async def proxy_request(request: Request, aleph_alpha_path: str, transform_body: Callable[[Dict[str, Any]], Dict[str, Any]] = None):
    headers = dict(request.headers.items())
    headers.pop("content-length", None)
    
    async with httpx.AsyncClient() as client:
        aleph_alpha_url = f"{ALEPH_ALPHA_API_BASE}{aleph_alpha_path}"
        body = await request.body()
        
        if transform_body:
            body = json.dumps(transform_body(await request.json()))
        
        response = await client.post(aleph_alpha_url, content=body, headers=headers) if body else await client.post(aleph_alpha_url, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        if request.query_params.get("stream") == "true":
            return StreamingResponse(
                response.aiter_bytes(),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(content=response.json())

def transform_body_completions(body: Dict[str, Any]) -> Dict[str, Any]:
    mappings = {
        "max_tokens": "maximum_tokens",
        "stop": "stop_sequences",
        "logprobs": "log_probs"
    }
    for old_key, new_key in mappings.items():
        if old_key in body:
            body[new_key] = body.pop(old_key)
    return body

async def transform_embeddings(request: Request) -> JSONResponse:
    body = await request.json()
    aleph_alpha_body = {
        "model": body.get("model", "luminous-base"),
        "prompt": body["input"],
        "layers": [-1],
        "pooling": ["mean"],
    }
    
    headers = dict(request.headers.items())
    headers.pop("content-length", None)
    
    async with httpx.AsyncClient() as client:
        aleph_alpha_url = f"{ALEPH_ALPHA_API_BASE}/embed"
        response = await client.post(aleph_alpha_url, json=aleph_alpha_body, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        aleph_alpha_data = response.json()
        openai_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": aleph_alpha_data["embeddings"]["layer_40"]["mean"],
                    "index": 0,
                }
            ],
            "model": aleph_alpha_data["model_version"],
            "usage": {
                "prompt_tokens": aleph_alpha_data["num_tokens_prompt_total"],
                "total_tokens": aleph_alpha_data["num_tokens_prompt_total"],
            },
        }
        return JSONResponse(content=openai_response)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request(request, "/chat/completions")

@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy_request(request, "/complete", transform_body=transform_body_completions)

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await transform_embeddings(request)
