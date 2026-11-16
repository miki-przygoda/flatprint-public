import io

from PIL import Image

from flatprint.backends.base import GenOutput
from flatprint.config import GenJob
from flatprint.runner import run
from flatprint.store import Shard

JOBS = [GenJob("flat", "m", "fake", "p", s, 8, 8, None, None) for s in range(5)]


class FakeBackend:
    name = "fake"

    def __init__(self, fail_seeds=()):
        self.fail_seeds = set(fail_seeds)
        self.calls = 0

    def generate(self, job):
        self.calls += 1
        if job.seed in self.fail_seeds:
            raise RuntimeError(f"boom {job.seed}")
        b = io.BytesIO()
        Image.new("RGB", (job.width, job.height), (job.seed, 0, 0)).save(b, "PNG")
        return GenOutput(b.getvalue(), "png", "cpu")


def test_run_then_resume_skips_done(data_root):
    shard = Shard(data_root, "mac")
    s1 = run(JOBS, shard, FakeBackend(), limit=2)
    assert (s1.done, s1.skipped) == (2, 0)
    b = FakeBackend()
    s2 = run(JOBS, Shard(data_root, "mac"), b)          # fresh Shard = simulated restart
    assert (s2.done, s2.skipped, s2.failed) == (3, 2, 0)
    assert b.calls == 3
    assert len(Shard(data_root, "mac").records()) == 5


def test_failures_are_logged_and_run_continues(data_root):
    shard = Shard(data_root, "mac")
    s = run(JOBS, shard, FakeBackend(fail_seeds={1, 3}))
    assert (s.done, s.failed) == (3, 2)
    assert len((shard.dir / "errors.jsonl").read_text().splitlines()) == 2


def test_too_many_consecutive_failures_aborts(data_root):
    s = run(JOBS, Shard(data_root, "mac"), FakeBackend(fail_seeds=range(5)), max_consecutive_failures=2)
    assert s.failed == 2 and s.done == 0


def test_record_fields_come_from_job(data_root):
    shard = Shard(data_root, "mac")
    run(JOBS[:1], shard, FakeBackend())
    r = shard.records()[0]
    assert (r.id, r.set, r.seed, r.device, r.backend) == (JOBS[0].key(), "flat", 0, "cpu", "fake")


def _two_prompt_jobs():
    return [GenJob("flat", "m", "fake", p, s, 8, 8, None, None) for p in ("black", "white") for s in range(3)]


def test_limit_takes_jobs_round_robin_across_prompts(data_root):
    shard = Shard(data_root, "mac")
    run(_two_prompt_jobs(), shard, FakeBackend(), limit=2)
    assert sorted(r.prompt for r in shard.records()) == ["black", "white"]


def test_time_budget_stops_before_starting_a_job_past_the_deadline(data_root):
    ticks = iter([0, 4, 8, 12, 16, 20, 24])         # start, then one reading before each job
    s = run(JOBS, Shard(data_root, "mac"), FakeBackend(), max_seconds=10, clock=lambda: next(ticks))
    assert s.done == 2 and s.out_of_time
    s2 = run(JOBS, Shard(data_root, "mac"), FakeBackend())
    assert s2.done == 3 and not s2.out_of_time       # the rest resumes later
