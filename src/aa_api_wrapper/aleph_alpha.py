import os
from aleph_alpha_client import Client, CompletionRequest, EmbeddingRequest, Prompt
from aleph_alpha_client.embedding import (
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)
from fastapi import Request
from openai.types.completion_create_params import CompletionCreateParams
from typing import Literal

from openai.types.embedding_create_params import EmbeddingCreateParams

from aa_api_wrapper.http import unpack_bearer_token


AaFinishReason = Literal["end_of_text", "maximum_tokens"] | None


def init_client(token: str) -> Client:
    aa_client = Client(
        host=os.environ.get("ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com"),
        nice=True,
        token=token,
    )
    return aa_client


def init_from_request(request: Request):
    token = unpack_bearer_token(request)
    return init_client(token)


def create_semantic_embedding_requests(
    embedding_params: EmbeddingCreateParams,
) -> list[SemanticEmbeddingRequest]:
    embedding_inputs = embedding_params["input"]

    if isinstance(embedding_inputs, str):
        embedding_inputs = [embedding_inputs]

    return [
        SemanticEmbeddingRequest(
            prompt=Prompt.from_text(embedding_input),  # type: ignore
            representation=SemanticRepresentation.Document,
        )
        for embedding_input in embedding_inputs
    ]


def create_embedding_requests(
    embedding_params: EmbeddingCreateParams,
) -> list[EmbeddingRequest]:
    embedding_inputs = embedding_params["input"]

    if isinstance(embedding_inputs, str):
        embedding_inputs = [embedding_inputs]

    return [
        EmbeddingRequest(
            prompt=Prompt.from_text(embedding_input),  # type: ignore
            layers=[-1],
            pooling=["last_token"],
        )
        for embedding_input in embedding_inputs
    ]


def create_completion_request(
    completion_params: CompletionCreateParams,
) -> CompletionRequest:
    text = completion_params["prompt"]
    assert isinstance(text, str), "Lists of prompts are currently not supported"
    prompt = Prompt.from_text(text)

    stop = completion_params.get("stop")
    stop_sequences = stop if isinstance(stop, list) else [stop] if stop else None

    return CompletionRequest(
        prompt=prompt,
        maximum_tokens=completion_params.get("max_tokens"),
        temperature=completion_params.get("temperature") or 0.0,
        top_p=completion_params.get("top_p") or 0.0,
        presence_penalty=completion_params.get("presence_penalty") or 0.0,
        frequency_penalty=completion_params.get("frequency_penalty") or 0.0,
        best_of=completion_params.get("best_of"),
        n=completion_params.get("n") or 1,
        # logit_bias=completion_params.get("logit_bias") or None,
        log_probs=completion_params.get("logprobs") or None,
        stop_sequences=stop_sequences,
        echo=completion_params.get("echo") or False,
    )
