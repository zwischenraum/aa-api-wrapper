import openai
from dotenv import load_dotenv

from openai.types.chat import ChatCompletionMessageParam

from aa_api_wrapper.settings import get_settings

load_dotenv()

settings = get_settings()

openai.base_url = "http://localhost:8000/v1/"
openai.api_key = settings.aa_token.get_secret_value() if settings.aa_token else None
print(openai.api_key)


def test_embeddings():
    response = openai.embeddings.create(input="Apple", model="luminous-base")
    print("Embeddings test:", response)


def test_multi_embeddings():
    response = openai.embeddings.create(
        input=["Apple", "Banana"], model="luminous-base"
    )
    print("Embeddings test:", response)


def test_chat_completions():
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    model = settings.aa_chat_model if settings.aa_chat_model else None
    response = openai.chat.completions.create(
        messages=messages, model=model
    )
    print("Chat completions test:", response)


def test_completions():
    response = openai.completions.create(
        model="luminous-base-control",
        prompt="What's the capital of Germany?",
        n=1,
        temperature=0.6,
        max_tokens=50,
    )
    print("Completions test:", response)


def test_models():
    response = openai.models.list()
    print("Models test:", response)


if __name__ == "__main__":
    test_models()
    print("-" * 20)
    test_embeddings()
    print("-" * 20)
    test_multi_embeddings()
    print("-" * 20)
    test_chat_completions()
    print("-" * 20)
    test_completions()
