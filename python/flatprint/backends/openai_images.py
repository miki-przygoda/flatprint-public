"""OpenAI Images API. The seed is recorded but not supported by the API."""

import base64
import os

from flatprint.backends.base import GenOutput, sniff_ext
from flatprint.backends.http import post_json_with_retry
from flatprint.config import GenJob

URL = "https://api.openai.com/v1/images/generations"


class OpenAIImagesBackend:
    name = "openai"

    def __init__(self, model: str, options: dict, client=None):
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self.options = options
        if client is None:
            import httpx
            client = httpx.Client()
        self.client = client

    def generate(self, job: GenJob) -> GenOutput:
        body = {"model": self.model, "prompt": job.prompt, "size": f"{job.width}x{job.height}", "n": 1,
                **self.options.get("request", {})}
        resp = post_json_with_retry(self.client, URL, json=body,
                                    headers={"Authorization": f"Bearer {self.key}"})
        data = base64.b64decode(resp["data"][0]["b64_json"])
        return GenOutput(data, sniff_ext(data), "api", {"revised_prompt": resp["data"][0].get("revised_prompt")})
