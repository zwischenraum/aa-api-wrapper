import json
from typing import cast

import httpx
from aleph_alpha_client import Client
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from openai._streaming import Stream
from openai.types import (
    CompletionCreateParams,
    EmbeddingCreateParams,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aa_api_wrapper.aleph_alpha import (
    create_completion_request,
    create_embedding_requests,
    create_semantic_embedding_requests,
)
from aa_api_wrapper.client import ManualClient
from aa_api_wrapper.http import (
    prepare_headers,
)
from aa_api_wrapper.settings import get_settings

settings = get_settings()

manual_client = ManualClient(
    settings.aleph_alpha_api_base,
)


async def proxy_request(
    request: Request,
    aleph_alpha_path: str,
) -> ChatCompletion | StreamingResponse | Stream[ChatCompletionChunk]:
    headers = prepare_headers(request)
    body = await request.body()

    json_body = json.loads(body)

    json_body.pop("n", None)
    json_body.pop("top_p", None)

    body = json.dumps(json_body).encode()

    should_stream = json_body.get("stream", False)

    # # Save the request to a file with .http extension
    # with open("request.http", "w") as file:
    #     file.write(f"{request.method} {request.url}\n")
    #     for key, value in headers.items():
    #         file.write(f"{key}: {value}\n")
    #     if body:
    #         file.write(f"\n{body.decode()}")

    if not should_stream:
        try:
            response = await manual_client.request(
                "POST", aleph_alpha_path, headers=headers, content=body
            )
        except httpx.HTTPStatusError as e:
            print(e.response.text)
            print(json.dumps(json_body, indent=2))
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.text
            )
        return ChatCompletion.model_validate(obj=response.json())

    return StreamingResponse(
        await manual_client.stream(
            "POST", aleph_alpha_path, headers=headers, content=body
        ),
        media_type="application/json",
    )


def proxy_semantic_embeddings(
    embedding_params: EmbeddingCreateParams, aa_client: Client, model: str
) -> list[list[float]]:
    embedding_vectors: list[list[float]] = []
    aa_requests = create_semantic_embedding_requests(embedding_params)
    for aa_request in aa_requests:
        aa_response = aa_client.semantic_embed(aa_request, model=model)
        embedding_vector = aa_response.embedding
        embedding_vectors.append(embedding_vector)
    return embedding_vectors


def proxy_regular_embeddings(
    embedding_params: EmbeddingCreateParams, aa_client: Client, model: str
) -> list[list[float]]:
    embedding_vectors: list[list[float]] = []
    aa_requests = create_embedding_requests(embedding_params)
    for aa_request in aa_requests:
        aa_response = aa_client.embed(aa_request, model=model)
        embeddings_dict = cast(
            dict[tuple[str, str], list[float]], aa_response.embeddings
        )
        embedding_vector = (
            embeddings_dict.values().__iter__().__next__()
        )  # just get the first value
        embedding_vectors.append(embedding_vector)
    return embedding_vectors


def proxy_completion(aa_client: Client, completion_params: CompletionCreateParams):
    aa_request = create_completion_request(completion_params)
    return aa_client.complete(aa_request, completion_params["model"])
