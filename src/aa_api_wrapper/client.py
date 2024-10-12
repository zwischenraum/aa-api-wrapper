from typing import Dict
import httpx


class ManualClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def request(
        self, method: str, path: str, headers: Dict[str, str], **kwargs
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response
