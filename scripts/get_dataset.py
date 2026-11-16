"""Download the flatprint dataset from Hugging Face (all of it, or a group of sets) as a ready data root.

    uv run python scripts/get_dataset.py fingerprint             # one group -> data/
    uv run python scripts/get_dataset.py scenes real laundered   # several groups
    uv run python scripts/get_dataset.py all                     # everything (~3.4 GB)
    uv run python scripts/get_dataset.py --sets showcase showcase-q85
    uv run python scripts/get_dataset.py --list                  # groups, their sets and sizes

The dataset (huggingface.co/datasets/mikiprzygoda/flatprint) stores one tar per set in archives/, next to
manifest.jsonl and the flagged lists. This fetches the manifest, the flagged lists and the chosen archives,
extracts them into DEST (default data/, the code's default data root, so no FLATPRINT_DATA is needed), deletes the
archives and checks every extracted image against its manifest sha256 (`flatprint verify --present`).

--archives DIR extracts tars already downloaded to DIR instead of downloading (no network).
Running it again with another group adds to the same DEST.
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

from flatprint import cli

REPO = "mikiprzygoda/flatprint"
ROOT = Path(__file__).resolve().parent.parent
GROUPS = {
    "fingerprint": ["flat-black-sizes", "flux-black", "flat-rgb", "flat-pilot", "vae-flat-rgb", "vae-flat-latent",
                    "vae-flat-latent-n01", "vae-flat-latent-n03", "vae-flat-latent-n06", "vae-noise",
                    "vae-random-latent"],
    "scenes": ["showcase", "adhoc", "z-scenes", "flux-scenes"],
    "real": ["phone-blank", "real-phone", "real-phone-dim"],
    "laundered": ["flat-black-half", "flat-black-sizes-q85", "flat-black-sizes-w736", "flat-black-sizes-w736-q85",
                  "flat-black-sizes-w564-q85", "showcase-q85", "showcase-w736", "showcase-w736-q85",
                  "showcase-w564-q85", "z-scenes-w736", "z-scenes-w736-q85", "z-scenes-w564-q85",
                  "real-phone-w736-q85", "real-phone-w564-q85", "real-phone-dim-w736-q85", "real-phone-dim-w564-q85"],
}
ABOUT = {
    "fingerprint": "flat blacks (4 sizes), FLUX blacks, flat colours, VAE-only controls: Fig. 1-2, signatures",
    "scenes": "Z-Image showcase + COCO scenes, FLUX scenes: the generated test images",
    "real": "own phone photos (lens covered, everyday, dim): the public real negatives",
    "laundered": "JPEG 85 / 736 / 564 px copies of the blacks, scenes and phone photos: robustness",
}
ALWAYS = ["manifest.jsonl", "flat-black-sizes-flagged/*", "flat-rgb-flagged/*"]


def wanted_sets(groups: list[str], sets: list[str]) -> list[str]:
    out = []
    for g in groups:
        if g == "all":
            out += [s for members in GROUPS.values() for s in members]
        elif g in GROUPS:
            out += GROUPS[g]
        else:
            raise SystemExit(f"unknown group '{g}' (groups: {', '.join(GROUPS)}, all)")
    return list(dict.fromkeys(out + sets))


def download(dest: Path, sets: list[str], repo: str) -> Path:
    from huggingface_hub import snapshot_download
    patterns = ALWAYS + [f"archives/{s}.tar" for s in sets]
    snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=patterns)
    return dest / "archives"


def extract(archives: Path, dest: Path, sets: list[str], keep: bool) -> None:
    missing = [s for s in sets if not (archives / f"{s}.tar").exists()]
    if missing:
        raise SystemExit(f"no archive for: {', '.join(missing)} (in {archives})")
    for s in sets:
        tar = archives / f"{s}.tar"
        with tarfile.open(tar) as t:
            t.extractall(dest, filter="data")         # rejects absolute paths, '..' and links out of dest
        print(f"extracted {s}")
        if not keep and archives.resolve() == (dest / "archives").resolve():
            tar.unlink()
    if not keep and (dest / "archives").is_dir() and not any((dest / "archives").iterdir()):
        (dest / "archives").rmdir()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("groups", nargs="*", help=f"{', '.join(GROUPS)} or all")
    ap.add_argument("--sets", nargs="+", default=[], help="individual sets, in addition to any groups")
    ap.add_argument("--dest", type=Path, default=ROOT / "data")
    ap.add_argument("--archives", type=Path, help="extract tars already in this folder; don't download")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--list", action="store_true", help="show the groups and exit")
    args = ap.parse_args(argv)
    if args.list or not (args.groups or args.sets):
        for g, members in GROUPS.items():
            print(f"{g:12} {ABOUT[g]}\n{'':12} {', '.join(members)}")
        print("all          every set above")
        return 0 if args.list else 2
    sets = wanted_sets(args.groups, args.sets)
    args.dest.mkdir(parents=True, exist_ok=True)
    if args.archives:
        for name in ("manifest.jsonl", "flat-black-sizes-flagged", "flat-rgb-flagged"):
            src = args.archives.parent / name if not (args.archives / name).exists() else args.archives / name
            if src.is_dir():
                shutil.copytree(src, args.dest / name, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, args.dest / name)
        archives = args.archives
    else:
        archives = download(args.dest, sets, args.repo)
    extract(archives, args.dest, sets, args.keep_archives)
    code = cli.main(["--data", str(args.dest), "verify", "--present"])
    if code == 0:
        hint = "" if args.dest.resolve() == (ROOT / "data").resolve() else f"export FLATPRINT_DATA={args.dest}\n"
        print(f"\nready: {len(sets)} sets in {args.dest}\n{hint}", end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
