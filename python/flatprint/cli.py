"""flatprint command line."""

import argparse
import os
import socket
import sys
from collections import Counter
from pathlib import Path

from flatprint.backends import get_backend
from flatprint.config import ConfigError, expand_jobs, load_config
from flatprint.runner import run
from flatprint.store import Shard, load_manifest, merge, resolve


def default_host() -> str:
    return socket.gethostname().split(".")[0].lower()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flatprint")
    p.add_argument("--data", type=Path, default=Path(os.environ.get("FLATPRINT_DATA", "data")))
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run a generation config into this host's shard")
    g.add_argument("config", type=Path)
    g.add_argument("--limit", type=int, help="stop after this many new images")
    g.add_argument("--hours", type=float, help="stop starting new images after this long")
    g.add_argument("--per-prompt", type=int, help="override the config's seeds.per_prompt")
    g.add_argument("--host", default=default_host())

    q = sub.add_parser("prompt", help="generate one ad-hoc image from free text (kept, seed recorded)")
    q.add_argument("text")
    q.add_argument("--config", type=Path, required=True,
                   help="config to take model, backend, steps, guidance and options from")
    q.add_argument("--set", default="adhoc")
    q.add_argument("--seed", type=int, help="default: random, recorded in the manifest")
    q.add_argument("--size", help="WxH, default: the config's first size")
    q.add_argument("--host", default=default_host())

    i = sub.add_parser("import", help="import a folder of existing images (bytes unchanged)")
    i.add_argument("dir", type=Path)
    i.add_argument("--set", required=True)
    i.add_argument("--source", required=True)
    i.add_argument("--licence")
    i.add_argument("--link-parents", action="store_true")
    i.add_argument("--host", default=default_host())

    d = sub.add_parser("derive", help="apply an op chain to a set (normalise / launder)")
    d.add_argument("--from-set", required=True)
    d.add_argument("--ops", required=True, help='e.g. "crop:512,png" or "resize:0.75,jpeg:85"')
    d.add_argument("--to-set", required=True)
    d.add_argument("--host", default=default_host())

    m = sub.add_parser("meta", help="inspect file metadata (PNG text, EXIF, XMP, C2PA)")
    m.add_argument("--set")

    e = sub.add_parser("export", help="copy a set out of the store with readable names (for previewing)")
    e.add_argument("--set", required=True)
    e.add_argument("--to", type=Path, required=True)

    v = sub.add_parser("verify", help="check every stored file against its manifest sha256 (e.g. after a copy)")
    v.add_argument("--set")
    v.add_argument("--present", action="store_true",
                   help="only sets with at least one file on disk (a partial download); fail if none is")

    sub.add_parser("merge", help="combine all shards into data/manifest.jsonl")
    sub.add_parser("info", help="counts per set and model in the merged manifest")
    return p


def cmd_generate(args) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.per_prompt:
        from dataclasses import replace
        cfg = replace(cfg, seeds_per_prompt=args.per_prompt)
    jobs = expand_jobs(cfg)
    backend = get_backend(cfg.backend, cfg.model, cfg.options)
    s = run(jobs, Shard(args.data, args.host), backend, limit=args.limit, progress=True,
            max_seconds=args.hours * 3600 if args.hours else None)
    print(f"{cfg.name}: done={s.done} skipped={s.skipped} failed={s.failed} (of {len(jobs)})"
          + (" stopped: time budget reached" if s.out_of_time else ""))
    return 0 if s.failed == 0 else 1


def cmd_prompt(args) -> int:
    import secrets
    from flatprint.config import GenJob
    try:
        cfg = load_config(args.config)
        w, h = (int(v) for v in args.size.lower().split("x")) if args.size else cfg.sizes[0]
    except (ConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    job = GenJob(args.set, cfg.model, cfg.backend, args.text, seed, w, h, cfg.steps, cfg.guidance)
    shard = Shard(args.data, args.host)
    s = run([job], shard, get_backend(cfg.backend, cfg.model, cfg.options))
    if s.done:
        rec = next(r for r in reversed(shard.records()) if r.id == job.key())
        print(f"{rec.id}  seed={seed}  {w}x{h}  -> {shard.dir / rec.file}")
    elif s.skipped:
        print(f"{job.key()} already exists (same prompt, seed and size)")
    else:
        print(f"failed; see {shard.dir / 'errors.jsonl'}", file=sys.stderr)
    return 0 if s.failed == 0 else 1


def cmd_import(args) -> int:
    from flatprint.importer import import_dir
    known = {r.id for r in load_manifest(args.data)} if args.link_parents else None
    s = import_dir(args.dir, Shard(args.data, args.host), set=args.set, source=args.source,
                   licence=args.licence, link_parents=args.link_parents, known_ids=known)
    print(f"import {args.set}: done={s.done} skipped={s.skipped} failed={s.failed}")
    return 0 if s.failed == 0 else 1


def cmd_derive(args) -> int:
    from flatprint.derive import derive_set
    recs = [r for r in load_manifest(args.data) if r.set == args.from_set]
    if not recs:
        print(f"no records in set '{args.from_set}' (did you run merge?)", file=sys.stderr)
        return 2
    s = derive_set(recs, args.data, Shard(args.data, args.host), args.ops, args.to_set, progress=True)
    print(f"derive {args.to_set}: done={s.done} skipped={s.skipped} failed={s.failed}")
    return 0


def cmd_meta(args) -> int:
    import json
    from flatprint.metadata import inspect_bytes
    from flatprint.store import resolve
    recs = [r for r in load_manifest(args.data) if args.set is None or r.set == args.set]
    counts: dict[str, Counter] = {}
    with open(args.data / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            info = inspect_bytes(resolve(args.data, r).read_bytes())
            f.write(json.dumps({"id": r.id, "set": r.set, **info}) + "\n")
            c = counts.setdefault(r.set, Counter())
            c["n"] += 1
            for k in ("c2pa", "xmp", "exif"):
                c[k] += info[k]
            c["text"] += bool(info["text_keys"])
    for set_, c in sorted(counts.items()):
        print(f"{set_:<28} n={c['n']:<6} c2pa={c['c2pa']:<6} xmp={c['xmp']:<6} exif={c['exif']:<6} text={c['text']}")
    return 0


def _slug(text: str, words: int = 8) -> str:
    import re
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:words]) or "untitled"


def cmd_export(args) -> int:
    """Copy one set into a folder per prompt, files named by seed. The store stays the source of truth."""
    import shutil
    from flatprint.store import resolve
    recs = [r for r in load_manifest(args.data) if r.set == args.set]
    if not recs:
        print(f"no records in set '{args.set}' (did you run merge?)", file=sys.stderr)
        return 2
    prompts = list(dict.fromkeys(r.prompt or "" for r in sorted(recs, key=lambda r: r.created_at or "")))
    if len({_slug(p) for p in prompts}) < len(prompts):
        # Prompts share their opening words (e.g. the exact-RGB template): name folders by what differs.
        import os
        words = [p.split() for p in prompts]
        shared = len(os.path.commonprefix(words))
        prompts_named = {p: " ".join(w[shared:]) for p, w in zip(prompts, words)}
    else:
        prompts_named = {p: p for p in prompts}
    folder = {p: f"{i:02d}-{_slug(prompts_named[p])}" for i, p in enumerate(prompts, 1)}
    # A seed repeats in a folder when a set has several sizes (or derived variants): widen the name until unique.
    seen = Counter((r.prompt or "", r.seed) for r in recs)
    sized = Counter((r.prompt or "", r.seed, r.width, r.height) for r in recs)
    for r in recs:
        stem = f"seed-{r.seed}"
        if seen[(r.prompt or "", r.seed)] > 1:
            stem += f"-{r.width}x{r.height}"
            if sized[(r.prompt or "", r.seed, r.width, r.height)] > 1:
                stem += f"-{r.id}"
        dest = args.to / folder[r.prompt or ""] / f"{stem}.{Path(r.file).suffix.lstrip('.')}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve(args.data, r), dest)
    (args.to / "prompts.txt").write_text("".join(f"{folder[p]}: {p}\n" for p in prompts), encoding="utf-8")
    print(f"exported {len(recs)} images in {len(prompts)} folders to {args.to}")
    return 0


def cmd_merge(args) -> int:
    recs, warnings = merge(args.data)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"merged {len(recs)} records")
    return 0


def cmd_verify(args) -> int:
    import hashlib
    recs = [r for r in load_manifest(args.data) if args.set is None or r.set == args.set]
    if not recs:
        print(f"nothing to verify: no records in {Path(args.data) / 'manifest.jsonl'}"
              + (f" for set '{args.set}'" if args.set else "") + " (incomplete download, or run merge?)",
              file=sys.stderr)
        return 1
    if args.present:
        on_disk = {r.set for r in recs if resolve(args.data, r).exists()}
        skipped = sorted({r.set for r in recs} - on_disk)
        recs = [r for r in recs if r.set in on_disk]
        if skipped:
            print(f"skipped {len(skipped)} set{'s' if len(skipped) > 1 else ''} not on disk: {', '.join(skipped)}")
        if not recs:
            print("nothing to verify: no set has any file on disk", file=sys.stderr)
            return 1
    same = differ = missing = 0
    for r in recs:
        p = resolve(args.data, r)
        if not p.exists():
            missing += 1
            print(f"missing {r.id} {r.file}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != r.sha256:
            differ += 1
            print(f"differs {r.id} {r.file}")
        else:
            same += 1
    print(f"{len(recs)} checked: {same} identical, {differ} differ, {missing} missing")
    return 1 if differ or missing else 0


def cmd_info(args) -> int:
    counts = Counter((r.set, r.model) for r in load_manifest(args.data))
    for (set_, model), n in sorted(counts.items()):
        print(f"{set_:<28} {model:<48} {n:>7}")
    return 0


COMMANDS = {"generate": cmd_generate, "prompt": cmd_prompt, "import": cmd_import, "derive": cmd_derive, "meta": cmd_meta, "export": cmd_export, "verify": cmd_verify, "merge": cmd_merge, "info": cmd_info}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
