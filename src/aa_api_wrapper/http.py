from typing import Dict, Any
from fastapi import Request


def prepare_headers(request: Request) -> Dict[str, str]:
    headers = dict(request.headers.items())
    headers.pop("content-length", None)
    return headers


def transform_body_completions(body: Dict[str, Any]) -> Dict[str, Any]:
    mappings = {
        "max_tokens": "maximum_tokens",
        "stop": "stop_sequences",
        "logprobs": "log_probs",
    }
    return {mappings.get(k, k): v for k, v in body.items()}


def unpack_bearer_token(request: Request) -> str:
    return request.headers["Authorization"].split("Bearer ")[1]
