"""Resumable generation loop: skip jobs already in the shard, log failures, keep going."""

import io
import time
from dataclasses import dataclass
from itertools import chain, zip_longest

from PIL import Image
from tqdm import tqdm

from flatprint.config import GenJob
from flatprint.store import Shard


@dataclass
class RunSummary:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    out_of_time: bool = False


def interleave(jobs: list[GenJob]) -> list[GenJob]:
    """Round-robin across prompts, so a capped or timed run covers every prompt evenly."""
    by_prompt: dict[str, list[GenJob]] = {}
    for j in jobs:
        by_prompt.setdefault(j.prompt, []).append(j)
    return [j for j in chain.from_iterable(zip_longest(*by_prompt.values())) if j is not None]


def run(jobs: list[GenJob], shard: Shard, backend, limit: int | None = None,
        max_consecutive_failures: int = 10, progress: bool = False,
        max_seconds: float | None = None, clock=time.monotonic) -> RunSummary:
    s = RunSummary()
    streak = 0
    start = clock()
    todo = [j for j in interleave(jobs) if not shard.has(j.key())]
    s.skipped = len(jobs) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    for job in tqdm(todo, disable=not progress, desc="generate"):
        if max_seconds is not None and clock() - start >= max_seconds:
            s.out_of_time = True  # stop between images; the rest resumes on the next run
            break
        try:
            out = backend.generate(job)
            with Image.open(io.BytesIO(out.data)) as im:
                width, height = im.size  # the real size: APIs may ignore the requested one
        except Exception as e:  # noqa: BLE001 — log and continue; abort on a streak
            shard.log_error(job.key(), f"{type(e).__name__}: {e}")
            s.failed += 1
            streak += 1
            if streak >= max_consecutive_failures:
                break
            continue
        streak = 0
        shard.add({"id": job.key(), "set": job.set, "model": job.model, "backend": job.backend,
                   "prompt": job.prompt, "seed": job.seed, "width": width, "height": height,
                   "steps": job.steps, "guidance": job.guidance, "device": out.device,
                   "extra": {**out.extra, "requested_size": [job.width, job.height]}},
                  out.data, out.ext)
        s.done += 1
    return s
