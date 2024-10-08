import json
import os
from typing import Callable, Dict, Any, Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .client import AlephAlphaClient
from .utils import prepare_headers, transform_body_completions

client = AlephAlphaClient(os.environ.get("ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com"))

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

async def chat_completions_handler(request: Request):
    return await proxy_request(request, "/chat/completions")

async def completions_handler(request: Request):
    return await proxy_request(request, "/complete", transform_body=transform_body_completions)

async def embeddings_handler(request: Request):
    return await transform_embeddings(request)
