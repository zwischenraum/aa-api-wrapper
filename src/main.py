import json
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

load_dotenv()

# Replace with your API base URL
ALEPH_ALPHA_API_BASE = os.environ.get("ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com")


async def proxy_request(request: Request, aleph_alpha_path: str, transform_body=None):
    headers = dict(request.headers.items())
    headers.pop("content-length", None)
    async with httpx.AsyncClient() as client:
        aleph_alpha_url = f"{ALEPH_ALPHA_API_BASE}{aleph_alpha_path}"
        body = await request.body()
        if transform_body:
            body = json.dumps(transform_body(await request.json()))
        
        if body:
            response = await client.post(aleph_alpha_url, content=body, headers=headers)
        else:
            response = await client.post(aleph_alpha_url, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        if "stream" in request.query_params and request.query_params["stream"] == "true":
            return StreamingResponse(
                (chunk async for chunk in response.aiter_bytes()),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(content=response.json())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request(request, "/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    def transform_body_completions(body):
        if "max_tokens" in body:
            body["maximum_tokens"] = body.pop("max_tokens")
        if "stop" in body:
            body["stop_sequences"] = body.pop("stop")
        if "logprobs" in body:
            body["log_probs"] = body.pop("logprobs")
        return body

    return await proxy_request(request, "/complete", transform_body=transform_body_completions)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    # OpenAI embeddings requests use a different structure than Aleph Alpha's /embed endpoint.
    # Here, we transform the request body to be compatible with Aleph Alpha.
    body = await request.json()
    aleph_alpha_body = {
        "model": body.get("model", "luminous-base"),  # Default to luminous-base
        "prompt": body["input"],
        "layers": [-1],  # Use the last layer by default
        "pooling": ["mean"],  # Use mean pooling by default
    }
    headers = dict(request.headers.items())
    headers.pop("content-length", None)
    async with httpx.AsyncClient() as client:
        aleph_alpha_url = f"{ALEPH_ALPHA_API_BASE}/embed"
        response = await client.post(
            aleph_alpha_url, json=aleph_alpha_body, headers=headers
        )

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
