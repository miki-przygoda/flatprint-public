"""Gemini image generation via generateContent (the inlineData bytes are stored as returned)."""

import base64
import os

from flatprint.backends.base import GenOutput, sniff_ext
from flatprint.backends.http import post_json_with_retry
from flatprint.config import GenJob

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiImagesBackend:
    name = "gemini"

    def __init__(self, model: str, options: dict, client=None):
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.model = model
        self.options = options
        if client is None:
            import httpx
            client = httpx.Client()
        self.client = client

    def generate(self, job: GenJob) -> GenOutput:
        body = {"contents": [{"parts": [{"text": job.prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"], **self.options.get("generation", {})}}
        resp = post_json_with_retry(self.client, URL.format(model=self.model), json=body,
                                    headers={"x-goog-api-key": self.key})
        for part in resp["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                data = base64.b64decode(part["inlineData"]["data"])
                return GenOutput(data, sniff_ext(data), "api", {"mime": part["inlineData"].get("mimeType")})
        raise RuntimeError(f"no image in Gemini response for job {job.key()}")
