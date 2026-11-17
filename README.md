# flatprint

Code for a COMP1818 (Artificial Intelligence Applications, University of Greenwich, 2026–27) coursework project:
estimating an image generator's fingerprint from **flat images** (Z-Image-Turbo asked for a plain black image), and
testing whether it identifies that generator's content images, whether it is specific to that generator, and
whether it survives JPEG and social-media resizing.

- **Dataset:** [mikiprzygoda/flatprint on Hugging Face](https://huggingface.co/datasets/mikiprzygoda/flatprint):
  6,426 images in 34 sets (Z-Image-Turbo and FLUX.1-schnell generations, their laundered copies, VAE-only controls
  and the author's own phone photos).
- **This repo:** the generation and storage pipeline (`flatprint` package), the analysis notebooks and scripts
  behind every number and figure in the report, and the saved result logs.

## Getting started

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 (uv installs it if missing).

```bash
git clone https://github.com/miki-przygoda/flatprint-public.git
cd flatprint-public
uv sync --extra analysis      # core (store, laundering, metadata, CLI) + analysis libraries
uv run pytest                 # unit tests, no data or GPU needed
uv run flatprint --help
```

Optional extras:

| Extra | For | Install |
|---|---|---|
| `analysis` | marimo notebooks, figures, tests (matplotlib, scipy) | `uv sync --extra analysis` |
| `baselines` | the published detectors (Corvi 2023, AEROBLADE) | `uv sync --extra baselines` |
| `local` | generating images with diffusers (Z-Image-Turbo, FLUX.1-schnell) | `uv sync --extra local` |

## Getting the dataset

Download the whole dataset or just the group you need. The script fetches it, extracts it into `data/` (the code's
default data root, so nothing else needs setting) and checks every image against its sha256:

```bash
uv run python scripts/get_dataset.py fingerprint      # one group; add more names, or `all` (~3.4 GB)
uv run python scripts/get_dataset.py --list           # the groups and their sets
uv run flatprint info                                 # images per set and model
```

| Group | Size | Sets | Enough to run |
|---|---|---|---|
| *(none)* | 5 MB | `manifest.jsonl` only | `recheck.py`, `rescore_scan.py`, `fig_robustness.py`: the tables and Fig. 3 |
| `fingerprint` | 1.5 GB | flat blacks at 4 sizes, FLUX blacks, flat colours, VAE-only controls (11 sets) | Fig. 1–2 (`fig_fingerprint.py`), the `cleanup`, `resolutions` and `fingerprint` notebooks |
| `scenes` | 0.65 GB | showcase and COCO scenes (Z-Image), FLUX scenes (4 sets) | the generated test images, e.g. `baselines.py` |
| `real` | 0.36 GB | own phone photos: lens covered, everyday, dim (3 sets) | the public real images |
| `laundered` | 0.88 GB | JPEG 85, 736 and 564 px copies of the above (16 sets) | the robustness rows |

Running it again with another group adds to the same folder. For the manifest alone:
`uv run hf download mikiprzygoda/flatprint manifest.jsonl --repo-type dataset --local-dir data`.

<details>
<summary>Without the script</summary>

The images are stored as one `.tar` per set in `archives/`, next to `manifest.jsonl` (a few dozen files, so a full
download stays well inside the Hub's per-request rate limit).

```bash
hf download mikiprzygoda/flatprint --repo-type dataset --local-dir flatprint-data
for f in flatprint-data/archives/*.tar; do tar -xf "$f" -C flatprint-data; done   # -> flatprint-data/images/<set>/
export FLATPRINT_DATA=$PWD/flatprint-data
uv run flatprint verify       # every image against the manifest; add --present after a partial download
```

`tar` ships with macOS, Linux and Windows 10+. For some sets only, pass patterns such as
`--include "archives/showcase*.tar" --include "manifest.jsonl"`.
</details>

`manifest.jsonl` holds one record per image (set, model, prompt, seed, size, steps, guidance, device, the laundering
ops of derived copies and the parent image). See the dataset card for the full list of sets and licences.

**Not in the dataset:** the social-media images (Pinterest) used as real negatives are third-party and are not
redistributed. Results that use them can't be recomputed from the public data alone, so their outputs are kept in
`logs/results/` (below).

## Reproducing the report

All commands run from the repo root. They read `data/` (or `FLATPRINT_DATA` if set). Figures go to `report/` by
default; pass another path to keep them elsewhere.

```bash
uv sync --extra analysis

# Fig. 1 and 2: fingerprint tiles (Z-Image, FLUX), spectrum, sample-size curves (all from the dataset)
uv run python experiments/fig_fingerprint.py --extra flux-black:1024 --out outputs/figures

# Fig. 3: AUC per laundering step (numbers from logs/results/)
uv run python experiments/fig_robustness.py outputs/figures/fig-robustness.pdf

# Tables: re-print the AUCs from the saved scan (showcase-only and 512-px subsets included)
uv run python experiments/recheck.py logs/results/20261109-1303-scan.json \
    logs/results/20261109-1303-scan-sizematched.json logs/results/20261109-1303-scan-sharpness.json \
    logs/results/20261109-1303-scan-combined.json

# Size-matched z, sharpness and the combined model, re-computed from the saved scan
# (writes SCAN-sizematched/-sharpness/-combined/-combined-nosharp.json next to its input)
mkdir -p outputs/rescore && cp logs/results/20261109-1303-scan.json outputs/rescore/
uv run python experiments/rescore_scan.py outputs/rescore/20261109-1303-scan.json
```

The rescored files come out identical to the ones in `logs/results/`.

Steps that need the social-media negatives (not public), so they only run on the original store:

```bash
uv run python experiments/scale_scan.py logs/results/SCAN.json      # score + scale scan of every test image (slow)
uv run python experiments/metadata_baseline.py                      # metadata rules G (generator tag) and E (no EXIF)
uv run python experiments/sigma_sweep.py                            # residual sigma 1-8: strength / score AUC per group
uv run marimo edit experiments/results.py                           # browse the scored images
```

Published detectors (needs a GPU for reasonable speed; Corvi's `Grag2021_latent` weights from
[grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection), path in `CORVI_WEIGHTS`):

```bash
uv sync --extra baselines
uv run python experiments/baselines.py --sets showcase z-scenes flux-black flux-scenes ... --out logs/results/baselines.csv
uv run python experiments/baselines.py --report logs/results/baselines.csv
```

### Saved results (`logs/results/`)

| File | What |
|---|---|
| `20261109-1303-scan.json` | strength, score and the per-scale scan of every test image |
| `…-sizematched.json`, `…-sharpness.json` | the two scan statistics, re-scored |
| `…-combined.json`, `…-combined-nosharp.json` | the combined logistic model's out-of-fold P(Z-Image), with and without sharpness |
| `baselines.csv`, `baselines-report.txt` | Corvi 2023 and AEROBLADE scores and AUCs |
| `metadata-baseline.csv` | metadata rules G and E |
| `sigma-sweep.csv` | strength and score AUC for residual sigma 1, 2, 3, 5, 8 (sigma 3 = the report's setting) |

### Notebooks (`experiments/`, marimo)

| Notebook | What |
|---|---|
| `cleanup.py` | flags flat outputs that aren't solid black (shape, text, noisy grey) |
| `resolutions.py` | signatures per output size, fold, spectrum lattice, sample-size curves |
| `controls.py` | colours, synthetic noise, JPEG, half size, real and social baselines, verification |
| `fingerprint.py` | fingerprint per colour, decoder-only (VAE round trip) comparison |
| `results.py` | browser for scored results |
| `flat_images.py`, `combine_images.py` | early exploration |

Open one with `uv run marimo edit experiments/<notebook>.py`. Shared code: `fingerprint_lib.py` (residual, fold,
correlation, scan statistics, cross-validated logistic regression) and `cleanup_lib.py`.

## Regenerating the data

Generation needs the `local` extra and the model weights (Z-Image-Turbo is about 33 GB). The configs used for the
dataset are in `configs/`.

```bash
uv sync --extra local
export HF_HUB_CACHE="$HOME/flatprint-cache/hf"     # optional: keep model weights in one place
uv run flatprint generate configs/zimage-turbo-flat-black-sizes.toml
uv run flatprint merge                              # -> data/manifest.jsonl + data/index.csv
```

The scene configs read COCO 2017 captions from `data/coco/captions_val2017.json` (from the
[COCO 2017 annotations](https://cocodataset.org/#download)). Laundered copies are derived from stored sets, for example:

```bash
uv run flatprint derive --from-set showcase --ops jpeg:85 --to-set showcase-q85
uv run flatprint derive --from-set showcase --ops maxwidth:736,png --to-set showcase-w736
uv run flatprint derive --from-set showcase --ops maxwidth:736,jpeg:85 --to-set showcase-w736-q85
uv run flatprint derive --from-set showcase --ops maxwidth:564,jpeg:85 --to-set showcase-w564-q85
```

Long runs can go in a detached tmux session: `scripts/overnight.sh CONFIG --hours 8`. Each machine writes to its own
shard (`data/shards/<host>/`) and `flatprint merge` combines them. Image ids are a hash of everything that made the
image, so re-running a config skips finished images.

## Layout

```
python/flatprint/   package: config, prompts, store, backends (diffusers, VAE round trip, APIs), runner, ops, metadata, CLI
configs/            generation configs for every generated set in the dataset
experiments/        analysis notebooks and scripts; third_party/ = Corvi et al.'s network code
logs/results/       saved results behind the report's tables and figures
scripts/            get_dataset.py (download a group), dataset export, tmux run helpers
tests/              pytest
```

## Credits and licences

- Code: MIT (see `LICENSE`), except `experiments/third_party/`, which is copied unchanged from
  [grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection) (Apache-2.0).
- AEROBLADE is reimplemented from the [official code](https://github.com/jonasricker/aeroblade) (Ricker et al., CVPR 2024).
- Models: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
  [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
- Scene prompts: [COCO 2017](https://cocodataset.org/) captions (Lin et al., 2014).
