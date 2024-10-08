from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os

from .handlers import chat_completions_handler, completions_handler, embeddings_handler

app = FastAPI()

load_dotenv()

ALEPH_ALPHA_API_BASE = os.environ.get("ALEPH_ALPHA_API_BASE", "https://api.aleph-alpha.com")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await chat_completions_handler(request)

@app.post("/v1/completions")
async def completions(request: Request):
    return await completions_handler(request)

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await embeddings_handler(request)
