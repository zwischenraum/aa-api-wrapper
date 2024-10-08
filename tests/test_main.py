import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_aleph_alpha_client():
    with patch('src.handlers.AlephAlphaClient') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def headers():
    return {
        "Authorization": f"Bearer {os.getenv('AA_TOKEN')}"
    }

def test_chat_completions(client, mock_aleph_alpha_client, headers):
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Hello, how can I help you?"}}]}
    mock_aleph_alpha_client.request.return_value = mock_response

    response = client.post("/v1/chat/completions", json={
        "model": "luminous-base",
        "messages": [{"role": "user", "content": "Hello"}]
    }, headers=headers)

    assert response.status_code == 200
    assert "choices" in response.json()
    assert response.json()["choices"][0]["message"]["content"] == "Hello, how can I help you?"

def test_completions(client, mock_aleph_alpha_client, headers):
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"text": "Berlin is the capital of Germany."}]}
    mock_aleph_alpha_client.request.return_value = mock_response

    response = client.post("/v1/completions", json={
        "model": "luminous-base-control",
        "prompt": "What's the capital of Germany?",
        "max_tokens": 50
    }, headers=headers)

    print(response)
    assert response.status_code == 200
    assert "choices" in response.json()
    assert response.json()["choices"][0]["text"] == "Berlin is the capital of Germany."

def test_embeddings(client, mock_aleph_alpha_client, headers):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": {"layer_40": {"mean": [0.1, 0.2, 0.3]}},
        "model_version": "luminous-base",
        "num_tokens_prompt_total": 1
    }
    mock_aleph_alpha_client.request.return_value = mock_response

    response = client.post("/v1/embeddings", json={
        "model": "luminous-base",
        "input": "Apple"
    }, headers=headers)

    assert response.status_code == 200
    assert "data" in response.json()
    assert "embeddings" in response.json()["data"][0]
    assert response.json()["data"][0]["embeddings"]["layer_40"]["mean"] == [0.1, 0.2, 0.3]

def test_chat_completions_stream(client, mock_aleph_alpha_client, headers):
    mock_response = MagicMock()
    mock_response.aiter_bytes.return_value = iter([b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
                                                   b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'])
    mock_aleph_alpha_client.request.return_value = mock_response

    response = client.post("/v1/chat/completions", json={
        "model": "luminous-base",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    content = b"".join(response.iter_content())
    assert b'data: {"choices": [{"delta": {"content": "Hello"}}]}' in content
    assert b'data: {"choices": [{"delta": {"content": " world"}}]}' in content
