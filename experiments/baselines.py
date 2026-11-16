"""Existing detectors on our sets, run unchanged (no re-training, no tuning on our data).

- Corvi et al. 2023 (ICASSP): ResNet-50 trained on latent-diffusion images ("Grag2021_latent" weights from
  github.com/grip-unina/DMimageDetection). Score = the network's mean logit (> 0 = generated), as their main.py.
- AEROBLADE (Ricker et al. 2024, CVPR): reconstruct the image with a latent-diffusion autoencoder; score =
  minus the LPIPS distance (VGG, 2nd layer, spatial mean) between image and reconstruction, as their
  run_aeroblade.py defaults. Their three autoencoders (SD1, SD2, Kandinsky 2.1) give `aeroblade` = the max over
  them (their "max"); `aeroblade-zflux` uses the Z-Image VAE, which is byte-identical to the FLUX.1 VAE.

Every image is centre-cropped to at most CROP x CROP (a multiple of 8), the same input for every detector: the
phone photos are 12-24 MP, too big for a VAE in one pass, and a crop keeps the pixels as they are.

    HF_HUB_CACHE=~/flatprint-cache/hf uv run --extra baselines python experiments/baselines.py \
        --sets showcase z-scenes ... --out logs/results/baselines.csv [--detectors ...] [--limit N]
    uv run python experiments/baselines.py --report logs/results/baselines.csv
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from flatprint.dataset import read_ids  # noqa: E402
from flatprint.store import load_manifest, resolve  # noqa: E402

DATA = Path(os.environ.get("FLATPRINT_DATA", Path(__file__).parent.parent / "data"))
CORVI_WEIGHTS = Path(os.environ.get("CORVI_WEIGHTS", Path.home() / "flatprint-cache" / "weights" / "corvi" / "weights"
                                    / "Grag2021_latent" / "model_epoch_best.pth"))
FLAGGED = [DATA / "flat-black-sizes-flagged" / "flagged.md", DATA / "flat-rgb-flagged" / "flagged.md",
           DATA / "web-black-pinterest-flagged" / "flagged.md"]
CROP = 1024
AUTOENCODERS = {  # name: (repo, subfolder, kind, weights variant)
    # AEROBLADE loads the fp16 SD weights and upcasts the VAE to fp32 (pipe.upcast_vae()); same here.
    "aeroblade-sd1": ("CompVis/stable-diffusion-v1-1", "vae", "kl", "fp16"),
    # stabilityai/stable-diffusion-2-base is gone from the Hub; this community copy's VAE files have the same
    # sha256 as Manojb/stable-diffusion-2-base and sd2-community/stable-diffusion-2-1-base (checked 2026-11-12)
    "aeroblade-sd2": ("sd2-community/stable-diffusion-2-base", "vae", "kl", "fp16"),
    "aeroblade-kd21": ("kandinsky-community/kandinsky-2-1", "movq", "vq", None),   # run in fp32 (theirs: fp16)
    "aeroblade-zflux": ("Tongyi-MAI/Z-Image-Turbo", "vae", "kl", None),   # same file as the FLUX.1-schnell VAE
}
AEROBLADE_MAX = ["aeroblade-sd1", "aeroblade-sd2", "aeroblade-kd21"]   # their default set -> "aeroblade"
SOCIAL_REAL = ["web-black-pinterest", "real-social", "real-social-hard"]   # the 36 negatives of Tables 2-3
REAL_PREFIXES = ("phone-", "real-", "web-")                                 # rows marked [real] in the report


def centre_crop(img: Image.Image, size=CROP) -> Image.Image:
    w, h = img.size
    cw, ch = min(w, size) // 8 * 8, min(h, size) // 8 * 8
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


# ---- Corvi 2023 -------------------------------------------------------------------------------

def load_corvi(device):
    import torch
    from third_party.corvi_resnet_mod import resnet50
    model = resnet50(num_classes=1, gap_size=1, stride0=1)          # 'res50stride1', as get_method_here.py
    dat = torch.load(CORVI_WEIGHTS, map_location="cpu")
    model.load_state_dict(dat["model"])
    return model.to(device).eval()


def corvi_score(model, img, device) -> float:
    import torch
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.no_grad():
        out = model(((x - mean) / std)[None].to(device)).cpu().numpy()
    out = out[:, 0]
    return float(out.mean()) if out.ndim > 1 else float(out[0])


# ---- AEROBLADE --------------------------------------------------------------------------------

def load_ae(name, device):
    import torch
    from diffusers import AutoencoderKL, VQModel
    repo, sub, kind, variant = AUTOENCODERS[name]
    cls = VQModel if kind == "vq" else AutoencoderKL
    ae = cls.from_pretrained(repo, subfolder=sub, variant=variant, use_safetensors=True)
    return ae.to(device, dtype=torch.float32).eval()


def load_lpips(device):
    import warnings

    import lpips
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lpips.LPIPS(net="vgg", verbose=False).to(device).eval()


def aeroblade_score(ae, lp, img, device, seed=1) -> float:
    """-LPIPS layer 2 (VGG) between image and its AE reconstruction, as AEROBLADE's _PatchedLPIPS (spatial,
    no upsampling, then the mean)."""
    import lpips as lpips_mod
    import torch
    from diffusers import VQModel
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        enc = ae.encode(x * 2 - 1)
        if hasattr(enc, "latent_dist"):                                   # as diffusers' retrieve_latents
            lat = enc.latent_dist.sample(generator=torch.Generator(device="cpu").manual_seed(seed))
        else:
            lat = enc.latents
        if isinstance(ae, VQModel):
            rec = ae.decode(lat, force_not_quantize=True, return_dict=False)[0]
        else:
            rec = ae.decode(lat, return_dict=False)[0]
        rec = (rec / 2 + 0.5).clamp(0, 1)
        rec = torch.round(rec * 255) / 255                                # their reconstructions are saved as PNG
        a, b = lp.scaling_layer(x * 2 - 1), lp.scaling_layer(rec * 2 - 1)
        fa, fb = lp.net.forward(a), lp.net.forward(b)
        k = 1                                                             # "lpips_vgg_2" = the 2nd layer
        d = (lpips_mod.normalize_tensor(fa[k]) - lpips_mod.normalize_tensor(fb[k])) ** 2
        dist = lp.lins[k](d).mean()
    return -float(dist)


# ---- run --------------------------------------------------------------------------------------

def run(args):
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    flagged = set().union(*(read_ids(p) for p in FLAGGED if p.exists()))
    recs = [r for r in load_manifest(DATA) if r.set in set(args.sets) and r.id not in flagged
            and (r.parent_id is None or r.parent_id not in flagged)]
    if args.limit:
        by_set = {}
        for r in recs:
            by_set.setdefault(r.set, []).append(r)
        recs = [r for rs in by_set.values() for r in rs[:args.limit]]
    missing = [r for r in recs if not resolve(DATA, r).exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(recs)} images are not on this machine (e.g. {missing[0].file})")
    done = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {(row["id"], row["detector"]) for row in csv.DictReader(f)}
    new = not args.out.exists()
    print(f"{len(recs)} images, detectors {args.detectors}, device {device}, {len(done)} scores already in {args.out}")
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "set", "parent_id", "width", "height", "crop", "detector", "score"])
        for det in args.detectors:
            todo = [r for r in recs if (r.id, det) not in done]
            if not todo:
                continue
            t0 = time.time()
            if det == "corvi":
                model = load_corvi(device)
                score = lambda img: corvi_score(model, img, device)  # noqa: E731
            else:
                ae, lp = load_ae(det, device), load_lpips(device)
                score = lambda img: aeroblade_score(ae, lp, img, device)  # noqa: E731
            for i, r in enumerate(todo, 1):
                img = centre_crop(Image.open(resolve(DATA, r)).convert("RGB"))
                w.writerow([r.id, r.set, r.parent_id or "", r.width, r.height, f"{img.width}x{img.height}", det,
                            f"{score(img):.6g}"])
                if i % 25 == 0 or i == len(todo):
                    f.flush()
                    print(f"  {det}: {i}/{len(todo)}  {(time.time() - t0) / i:.2f} s/image", flush=True)
            del score
            if device == "mps":
                torch.mps.empty_cache()


def auc(pos, neg) -> float:
    """P(score of a positive > score of a negative), ties count half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def report(path: Path, negatives: list[str]):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    scores = {}
    for row in rows:
        scores.setdefault(row["detector"], {}).setdefault(row["set"], {})[row["id"]] = float(row["score"])
    per_id = {}                                                    # AEROBLADE "max" over its three AEs
    for det in AEROBLADE_MAX:
        for s, ids in scores.get(det, {}).items():
            for i, v in ids.items():
                per_id.setdefault((s, i), []).append(v)
    if per_id:
        scores["aeroblade"] = {}
        for (s, i), vs in per_id.items():
            if len(vs) == len(AEROBLADE_MAX):
                scores["aeroblade"].setdefault(s, {})[i] = max(vs)
    dets = [d for d in ["corvi", "aeroblade", "aeroblade-zflux", *AEROBLADE_MAX] if d in scores]
    sets = sorted({s for d in dets for s in scores[d]} - set(negatives))
    print(f"AUC vs {negatives} (n = {[len(scores[dets[0]].get(s, {})) for s in negatives]})")
    print("set".ljust(28) + "".join(d.rjust(17) for d in dets))
    for s in sets:
        line = s.ljust(28)
        for d in dets:
            neg = [v for n in negatives for v in scores[d].get(n, {}).values()]
            line += f"{auc(list(scores[d].get(s, {}).values()), neg):17.3f}"
        print(line + f"   (n={len(scores[dets[0]].get(s, {}))})" + ("  [real]" if s.startswith(REAL_PREFIXES) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=[])
    ap.add_argument("--detectors", nargs="+", default=["corvi", *AUTOENCODERS])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "logs" / "results" / "baselines.csv")
    ap.add_argument("--limit", type=int, default=0, help="first N images per set (smoke test)")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--negatives", nargs="+", default=SOCIAL_REAL)
    args = ap.parse_args()
    if args.report:
        report(args.report, args.negatives)
    else:
        run(args)


if __name__ == "__main__":
    main()
