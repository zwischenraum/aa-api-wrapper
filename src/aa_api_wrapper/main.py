from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os
import uvicorn

from aa_api_wrapper.handlers import (
    chat_completions_handler,
    completions_handler,
    embeddings_handler,
)


app = FastAPI()

load_dotenv()

ALEPH_ALPHA_API_BASE = os.environ.get(
    "ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com"
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await chat_completions_handler(request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await completions_handler(request)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await embeddings_handler(request)


def main():
    uvicorn.run("aa_api_wrapper.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
