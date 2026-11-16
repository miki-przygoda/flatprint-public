"""Content-addressed image store: one shard per host, JSONL manifests, merge."""

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Record:
    id: str
    set: str
    model: str
    backend: str
    file: str
    sha256: str
    host: str
    prompt: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    device: str | None = None
    created_at: str | None = None
    source: str | None = None
    licence: str | None = None
    parent_id: str | None = None
    ops: str | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Record":
        return cls(**json.loads(s))


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(" ", "_")


def _read_jsonl(path: Path) -> list[Record]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(Record.from_json(line))
        except (json.JSONDecodeError, TypeError):
            continue  # truncated line from an interrupted append
    return out


class Shard:
    def __init__(self, root: Path, host: str):
        self.root = Path(root)
        self.host = host
        self.dir = self.root / "shards" / host
        self.dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self.dir / "manifest.jsonl"
        self._terminate_partial_line()
        self._ids = {r.id for r in _read_jsonl(self._manifest)}

    def _terminate_partial_line(self) -> None:
        """After a crash mid-append, end the partial line so the next record starts clean."""
        if self._manifest.exists() and self._manifest.stat().st_size:
            with open(self._manifest, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")

    def has(self, id: str) -> bool:
        return id in self._ids

    def records(self) -> list[Record]:
        return _read_jsonl(self._manifest)

    def add(self, fields: dict, data: bytes, ext: str) -> Record:
        rel = Path("images") / fields["set"] / _slug(fields["model"]) / f"{fields['id']}.{ext}"
        dest = self.dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)  # atomic: no half-written image under the final name
        rec = Record(**{**fields, "file": rel.as_posix(), "sha256": hashlib.sha256(data).hexdigest(),
                        "host": self.host,
                        "created_at": fields.get("created_at") or datetime.now(timezone.utc).isoformat()})
        with open(self._manifest, "a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._ids.add(rec.id)
        return rec

    def log_error(self, id: str, message: str) -> None:
        with open(self.dir / "errors.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": id, "message": message,
                                "at": datetime.now(timezone.utc).isoformat()}) + "\n")


def merge(root: Path) -> tuple[list[Record], list[str]]:
    """Combine every shard's manifest into root/manifest.jsonl (paths relative to root).

    Records already in root/manifest.jsonl that don't live in a shard (e.g. a downloaded dataset, which has a
    manifest but no shards) are kept, so merging new shards into such a root adds to it instead of replacing it.
    """
    root = Path(root)
    by_id: dict[str, Record] = {r.id: r for r in _read_jsonl(root / "manifest.jsonl")
                                if not r.file.startswith("shards/")}
    warnings = []
    for shard_dir in sorted((root / "shards").glob("*")):
        for r in _read_jsonl(shard_dir / "manifest.jsonl"):
            r.file = (Path("shards") / shard_dir.name / r.file).as_posix()
            if r.id in by_id:
                if by_id[r.id].sha256 != r.sha256:
                    warnings.append(f"{r.id}: different pixels on {by_id[r.id].host} and {r.host}; "
                                    f"kept {by_id[r.id].host}")
                continue
            by_id[r.id] = r
    recs = list(by_id.values())
    (root / "manifest.jsonl").write_text("".join(r.to_json() + "\n" for r in recs), encoding="utf-8")
    write_index(root, recs)
    return recs, warnings


INDEX_COLUMNS = ["set", "model", "prompt", "seed", "width", "height", "host", "device",
                 "parent_id", "ops", "created_at", "id", "file"]


def write_index(root: Path, recs: list[Record]) -> None:
    """A spreadsheet-friendly view of the manifest, sorted by set, model, prompt and seed."""
    key = lambda r: (r.set, r.model, r.prompt or "", r.seed if r.seed is not None else -1, r.id)  # noqa: E731
    with open(Path(root) / "index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(recs, key=key):
            w.writerow(asdict(r))


def load_manifest(root: Path) -> list[Record]:
    return _read_jsonl(Path(root) / "manifest.jsonl")


def resolve(root: Path, rec: Record) -> Path:
    return Path(root) / rec.file
