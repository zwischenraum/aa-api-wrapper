import openai
from dotenv import load_dotenv
from os import getenv

load_dotenv()

openai.base_url = "http://localhost:8000/v1/"
openai.api_key = getenv("AA_TOKEN")

def test_embeddings():
    response = openai.embeddings.create(input="Apple", model="luminous-base")
    print("Embeddings test:", response)

def test_chat_completions():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = openai.chat.completions.create(messages=messages, model="luminous-base-control")
    print("Chat completions test:", response)

def test_completions():
    response = openai.completions.create(
        model="luminous-base-control",
        prompt="What's the capital of Germany?",
        n=1,
        temperature=0.6,
        max_tokens=50
    )
    print("Completions test:", response)

if __name__ == "__main__":
    test_embeddings()
    test_chat_completions()
    test_completions()
