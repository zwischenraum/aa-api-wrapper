import json
import os
from typing import Callable, Dict, Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

load_dotenv()

ALEPH_ALPHA_API_BASE = os.environ.get("ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com")

class AlephAlphaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def request(self, method: str, path: str, headers: Dict[str, str], **kwargs) -> httpx.Response:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response

client = AlephAlphaClient(ALEPH_ALPHA_API_BASE)

def prepare_headers(request: Request) -> Dict[str, str]:
    headers = dict(request.headers.items())
    headers.pop("content-length", None)
    return headers

def transform_body_completions(body: Dict[str, Any]) -> Dict[str, Any]:
    mappings = {
        "max_tokens": "maximum_tokens",
        "stop": "stop_sequences",
        "logprobs": "log_probs"
    }
    return {mappings.get(k, k): v for k, v in body.items()}

async def proxy_request(
    request: Request,
    aleph_alpha_path: str,
    transform_body: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
) -> StreamingResponse | JSONResponse:
    headers = prepare_headers(request)
    body = await request.body()
    
    if transform_body:
        body = json.dumps(transform_body(await request.json()))
    
    try:
        response = await client.request("POST", aleph_alpha_path, headers=headers, content=body)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    if request.query_params.get("stream") == "true":
        return StreamingResponse(
            response.aiter_bytes(),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(content=response.json())

async def transform_embeddings(request: Request) -> JSONResponse:
    body = await request.json()
    aleph_alpha_body = {
        "model": body.get("model", "luminous-base"),
        "prompt": body["input"],
        "layers": [-1],
        "pooling": ["mean"],
    }
    
    headers = prepare_headers(request)
    
    try:
        response = await client.request("POST", "/embed", headers=headers, json=aleph_alpha_body)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

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
