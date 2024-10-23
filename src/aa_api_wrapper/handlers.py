from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types import (
    Completion,
    CompletionCreateParams,
    CreateEmbeddingResponse,
    EmbeddingCreateParams,
)

from aa_api_wrapper.aleph_alpha import (
    init_from_request,
)

from aa_api_wrapper.openai import (
    CompletionCreateParamsAdapter,
    EmbeddingCreateParamsAdapter,
    create_completion_response,
    create_embedding_response,
)
from aa_api_wrapper.proxy import (
    proxy_completion,
    proxy_regular_embeddings,
    proxy_request,
    proxy_semantic_embeddings,
)
from aa_api_wrapper.settings import get_settings

settings = get_settings()


async def chat_completions_handler(
    request: Request,
) -> StreamingResponse | JSONResponse:
    return await proxy_request(request, "/chat/completions")


async def completions_handler(request: Request) -> Completion:
    aa_client = init_from_request(request)

    body = await request.json()
    completion_params: CompletionCreateParams = (
        CompletionCreateParamsAdapter.validate_python(body)
    )
    aa_response = proxy_completion(aa_client, completion_params)

    return create_completion_response(aa_response, completion_params["model"])


async def embeddings_handler(request: Request) -> CreateEmbeddingResponse:
    aa_client = init_from_request(request)

    body = await request.json()
    embedding_params: EmbeddingCreateParams = (
        EmbeddingCreateParamsAdapter.validate_python(body)
    )
    model = embedding_params["model"]

    if settings.use_semantic_embeddings:
        embedding_vectors = proxy_semantic_embeddings(
            embedding_params, aa_client, model
        )
    else:
        embedding_vectors = proxy_regular_embeddings(embedding_params, aa_client, model)

    return create_embedding_response(embedding_vectors=embedding_vectors, model=model)
