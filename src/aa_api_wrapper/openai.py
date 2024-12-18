from typing import Literal, cast

from aleph_alpha_client.completion import CompletionResponse
from openai.pagination import SyncPage
from openai.types import (
    Completion,
    CompletionChoice,
    CompletionCreateParams,
    CreateEmbeddingResponse,
    Model,
)
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding
from openai.types.embedding_create_params import EmbeddingCreateParams
from pydantic import TypeAdapter

from aa_api_wrapper.aleph_alpha import AaFinishReason

OpenAiFinishReason = Literal["stop", "length", "content_filter"]

CompletionCreateParamsAdapter = TypeAdapter(CompletionCreateParams)
EmbeddingCreateParamsAdapter = TypeAdapter(EmbeddingCreateParams)


def create_models_response(models: list) -> SyncPage[Model]:
    model_objects = [Model.model_validate(model) for model in models]
    return SyncPage(data=model_objects, object="list")


def create_embedding_response(
    embedding_vectors: list[list[float]],
    model: str,
) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        model=model,
        object="list",
        usage=create_empty_usage(),
        data=[
            Embedding(
                object="embedding",
                embedding=embedding_vector,
                index=0,
            )
            for embedding_vector in embedding_vectors
        ],
    )


def create_empty_usage() -> Usage:
    return Usage(prompt_tokens=-1, total_tokens=-1)


def create_completion_response(
    aa_response: CompletionResponse, model: str
) -> Completion:
    completion_result = aa_response.completions[0]
    completion = completion_result.completion
    assert completion is not None

    completion_choice = CompletionChoice(
        index=0,
        text=completion,
        finish_reason=_map_finish_reason(
            cast(AaFinishReason, completion_result.finish_reason)
        ),
    )
    return Completion(
        id="",
        choices=[completion_choice],
        created=0,
        model=model,
        object="text_completion",
    )


def _map_finish_reason(
    finish_reason: AaFinishReason,
) -> OpenAiFinishReason:
    if finish_reason == "end_of_text":
        return "stop"
    elif finish_reason == "maximum_tokens":
        return "length"
    else:
        raise ValueError(f"Unknown finish reason: {finish_reason}")
