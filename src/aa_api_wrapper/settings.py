from functools import cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    aa_token: SecretStr = Field(alias="AA_TOKEN")
    aleph_alpha_api_base: str = Field(alias="ALEPH_ALPHA_API_BASE")
    aa_chat_model: str = Field(alias="AA_CHAT_MODEL")
    use_semantic_embeddings: bool = Field(alias="USE_SEMANTIC_EMBEDDINGS")


@cache
def get_settings() -> Settings:
    settings = Settings(_env_file=".env")
    print(settings)
    return settings
