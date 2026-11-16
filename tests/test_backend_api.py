import base64
import json

import httpx
import pytest

from flatprint.backends.gemini_images import GeminiImagesBackend
from flatprint.backends.http import post_json_with_retry
from flatprint.backends.openai_images import OpenAIImagesBackend
from flatprint.config import GenJob

JPEG_WITH_C2PA = b"\xff\xd8\xff\xeb\x00\x10JUMBc2pa-manifest\xff\xd9"   # pretend APP11 C2PA
JOB = GenJob("closed", "gpt-image-1", "openai", "a plain black image", 0, 1024, 1024, None, None)


def client_returning(responses):
    calls = []

    def handler(request):
        calls.append(request)
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_retry_on_429_then_success():
    client, calls = client_returning([(429, {}), (503, {}), (200, {"ok": 1})])
    out = post_json_with_retry(client, "https://x/y", json={}, headers={}, sleep=lambda s: None)
    assert out == {"ok": 1} and len(calls) == 3


def test_retry_gives_up_and_does_not_retry_400():
    client, calls = client_returning([(400, {"error": "bad prompt"})])
    with pytest.raises(httpx.HTTPStatusError):
        post_json_with_retry(client, "https://x/y", json={}, headers={}, sleep=lambda s: None)
    assert len(calls) == 1


def test_openai_stores_bytes_exactly(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    b64 = base64.b64encode(JPEG_WITH_C2PA).decode()
    client, calls = client_returning([(200, {"data": [{"b64_json": b64}]})])
    out = OpenAIImagesBackend("gpt-image-1", {}, client=client).generate(JOB)
    assert out.data == JPEG_WITH_C2PA and out.ext == "jpg" and out.device == "api"
    sent = json.loads(calls[0].content)
    assert sent["model"] == "gpt-image-1" and sent["size"] == "1024x1024" and sent["prompt"] == JOB.prompt
    assert calls[0].headers["authorization"] == "Bearer sk-test"


def test_gemini_stores_bytes_exactly(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    b64 = base64.b64encode(JPEG_WITH_C2PA).decode()
    body = {"candidates": [{"content": {"parts": [{"text": "here"},
                                                  {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]}}]}
    client, calls = client_returning([(200, body)])
    out = GeminiImagesBackend("gemini-img", {}, client=client).generate(
        GenJob("closed", "gemini-img", "gemini", "p", 0, 1024, 1024, None, None))
    assert out.data == JPEG_WITH_C2PA and out.ext == "jpg"
    assert "gemini-img:generateContent" in str(calls[0].url)
    assert calls[0].headers["x-goog-api-key"] == "g-test"


def test_missing_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIImagesBackend("gpt-image-1", {})
