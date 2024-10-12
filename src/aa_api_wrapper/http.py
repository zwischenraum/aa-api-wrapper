from typing import Dict
from fastapi import Request


def prepare_headers(request: Request) -> Dict[str, str]:
    headers = dict(request.headers.items())
    headers.pop("content-length", None)
    return headers


def unpack_bearer_token(request: Request) -> str:
    return request.headers["Authorization"].split("Bearer ")[1]
