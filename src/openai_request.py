import openai
from dotenv import load_dotenv
from os import getenv

load_dotenv()

messages = [{"role": "user", "content": "hello world!"}]

openai.base_url = "http://localhost:8000/v1/"
openai.api_key = getenv("AA_TOKEN")

print(openai.embeddings.create(input="Apple", model="luminous-base"))
# print(openai.chat.completions.create(messages=messages, model="luminous-base-control"))
print(openai.completions.create(model="luminous-base-control", prompt="What's the capital of germany?", n=10, temperature=0.6, logprobs=1))
