import json
import os
from typing import Callable, Dict, Any, Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types import (
    Completion,
    CompletionCreateParams,
    CreateEmbeddingResponse,
    EmbeddingCreateParams,
)

from aa_api_wrapper.aleph_alpha import (
    create_completion_request,
    create_embedding_request,
    init_from_request,
)
from aa_api_wrapper.client import ManualClient
from aa_api_wrapper.http import (
    prepare_headers,
)

from aa_api_wrapper.openai import (
    CompletionCreateParamsAdapter,
    EmbeddingCreateParamsAdapter,
    create_completion_response,
    create_embedding_response,
)

manual_client = ManualClient(
    os.environ.get("ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com")
)


async def proxy_request(
    request: Request,
    aleph_alpha_path: str,
    transform_body: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> StreamingResponse | JSONResponse:
    headers = prepare_headers(request)
    body = await request.body()

    if transform_body:
        body = json.dumps(transform_body(await request.json()))

    try:
        response = await manual_client.request(
            "POST", aleph_alpha_path, headers=headers, content=body
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    if json.loads(body).get("stream"):
        return StreamingResponse(
            response.aiter_bytes(),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(content=response.json())


async def transform_embeddings(request: Request) -> CreateEmbeddingResponse:
    aa_client = init_from_request(request)

    body = await request.json()
    embedding_params: EmbeddingCreateParams = (
        EmbeddingCreateParamsAdapter.validate_python(body)
    )
    model = embedding_params["model"]

    aa_request = create_embedding_request(embedding_params)
    aa_response = aa_client.semantic_embed(aa_request, model=model)

    return create_embedding_response(
        embedding_vector=aa_response.embedding, model=model
    )


async def transform_complete(request: Request) -> Completion:
    aa_client = init_from_request(request)

    body = await request.json()
    completion_params: CompletionCreateParams = (
        CompletionCreateParamsAdapter.validate_python(body)
    )

    aa_request = create_completion_request(completion_params)
    aa_response = aa_client.complete(aa_request, completion_params["model"])

    return create_completion_response(aa_response, completion_params["model"])


async def chat_completions_handler(request: Request):
    return await proxy_request(request, "/chat/completions")


async def completions_handler(request: Request):
    return await transform_complete(request)


async def embeddings_handler(request: Request):
    return await transform_embeddings(request)
