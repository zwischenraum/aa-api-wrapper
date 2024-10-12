import json
import os

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
) -> StreamingResponse | JSONResponse:
    headers = prepare_headers(request)
    body = await request.body()

    json_body = json.loads(body)
    should_stream = json_body.get("stream", False)

    if not should_stream:
        try:
            response = await manual_client.request(
                "POST", aleph_alpha_path, headers=headers, content=body
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.text
            )
        return JSONResponse(content=response.json())

    return StreamingResponse(
        await manual_client.stream(
            "POST", aleph_alpha_path, headers=headers, content=body
        ),
        media_type="application/json",
    )


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
