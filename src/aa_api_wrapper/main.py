from fastapi import FastAPI, Request
import uvicorn

from aa_api_wrapper.handlers import (
    chat_completions_handler,
    completions_handler,
    embeddings_handler,
    models_handler,
)


app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await chat_completions_handler(request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await completions_handler(request)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await embeddings_handler(request)


@app.get("/v1/models")
async def models(request: Request):
    return await models_handler(request)


def main():
    uvicorn.run("aa_api_wrapper.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
