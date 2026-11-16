"""JSON POST with exponential backoff on rate limits and server errors."""

import time

import httpx

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def post_json_with_retry(client: httpx.Client, url: str, *, json: dict, headers: dict,
                         max_tries: int = 5, sleep=time.sleep) -> dict:
    for attempt in range(max_tries):
        resp = client.post(url, json=json, headers=headers, timeout=300)
        if resp.status_code in RETRY_STATUS and attempt < max_tries - 1:
            sleep(min(2 ** attempt * 2, 60))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")
