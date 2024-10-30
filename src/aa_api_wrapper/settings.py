from functools import cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    aleph_alpha_api_base: str = Field(alias="ALEPH_ALPHA_API_BASE")
    use_semantic_embeddings: bool = Field(alias="USE_SEMANTIC_EMBEDDINGS")
    http_timeout: int = Field(alias="HTTP_TIMEOUT", default=60 * 10)
    aa_token: SecretStr | None = Field(default=None, alias="AA_TOKEN")
    aa_chat_model: str | None = Field(default=None, alias="AA_CHAT_MODEL")


@cache
def get_settings() -> Settings:
    settings = Settings(_env_file=".env")
    print(settings)
    return settings
