from typing import Dict
import httpx

from aa_api_wrapper.settings import get_settings


class ManualClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _init_async_client(self):
        return httpx.AsyncClient(timeout=get_settings().http_timeout)

    async def request(
        self, method: str, path: str, headers: Dict[str, str], **kwargs
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        async with self._init_async_client() as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response

    async def stream(self, method: str, path: str, headers: Dict[str, str], **kwargs):
        url = f"{self.base_url}{path}"

        async def stream_generator():
            async with self._init_async_client() as client, client.stream(
                method, url, headers=headers, **kwargs
            ) as response:
                async for chunk in response.aiter_raw():
                    yield chunk

        return stream_generator()
