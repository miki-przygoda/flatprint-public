"""Run controls.py's scoring (sections 7 and 9, every run button pressed) once and save one row per picture.

    uv run python experiments/score_all.py OUT.json

OUT.json holds {"meta": thresholds and settings, "rows": [...]}; experiments/results.py browses it. It folds every
stored test picture, so it takes a while.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main(out: Path) -> None:
    sys.argv = [sys.argv[0], "--all"]
    import controls

    _, d = controls.app.run()
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "correlation_threshold": d["threshold"], "strength_threshold": d["strength_threshold"],
            "reference": controls.REF, "residual": d["mode_pick"].value, "sigma": d["sigma_pick"].value,
            "period": d["period_pick"].value, "holdout_per_group": d["holdout_pick"].value,
            "holdout_seed": d["holdout_seed"].value, "threshold_percentile": d["pct_pick"].value}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": d["verify_rows"]}), encoding="utf-8")
    print(f"{len(d['verify_rows'])} rows -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
