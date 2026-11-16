"""Figure: AUC of each detector at each laundering step (showcase Z scenes vs the 36 social-media real pictures).

Numbers from the result logs (logs/results/): single inputs from 20261109-1303-scan (-sizematched), the combined
model from 20261109-1303-scan-combined(-nosharp), all on the 32 showcase images (the 2 adhoc scenes left out so every
line uses the same pictures; experiments/recheck.py item 2), the existing detectors from baselines-report.txt (experiments/baselines.py,
run unchanged). Same 36 negatives throughout. Ours solid, existing detectors dotted with open markers.

    uv run python experiments/fig_robustness.py [out.pdf]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = ["native", "JPEG 85", "736 w", "736 w\n+ JPEG", "564 w\n+ JPEG"]
AUC = {  # label: (values per step, Okabe-Ito colour, marker, line style)
    "strength (as is)":       ([0.980, 0.923, 0.565, 0.567, 0.622], "#0072B2", "o", "-"),
    "scan z (size-matched)":  ([0.850, 0.719, 0.770, 0.742, 0.818], "#E69F00", "s", "-"),
    "combined, all four":     ([0.916, 0.740, 0.707, 0.682, 0.677], "#009E73", "^", "-"),
    "combined, no sharpness": ([0.928, 0.799, 0.763, 0.734, 0.682], "#CC79A7", "v", "--"),
}
BASELINES = {  # existing detectors, same groups
    "Corvi 2023":             ([0.734, 0.721, 0.319, 0.535, 0.470], "#D55E00", "D"),
    "AEROBLADE":              ([0.502, 0.487, 0.582, 0.565, 0.599], "#56B4E9", "P"),
    "AEROBLADE (Z/FLUX VAE)": ([0.756, 0.618, 0.349, 0.585, 0.595], "#000000", "X"),
}


def main(out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    x = range(len(STEPS))
    for label, (vals, colour, marker, ls) in AUC.items():
        ax.plot(x, vals, color=colour, marker=marker, linestyle=ls, linewidth=1.2, markersize=4, label=label)
    for label, (vals, colour, marker) in BASELINES.items():
        ax.plot(x, vals, color=colour, marker=marker, linestyle=":", linewidth=1.0, markersize=4,
                markerfacecolor="white", label=label)
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1)
    ax.text(4, 0.505, "coin flip", ha="right", va="bottom", fontsize=6.5, color="0.3")
    ax.set_xticks(list(x), STEPS)
    ax.set_ylim(0.25, 1.0)
    ax.set_ylabel("AUC")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "report" / "fig-robustness.pdf")
