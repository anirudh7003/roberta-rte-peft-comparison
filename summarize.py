"""Aggregate the per-seed JSON records written by train.py."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in Path("results").glob("*.json")]
    if not records:
        raise SystemExit("No result files found. Run train.py first.")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)

    print("method     seeds  accuracy (mean ± std)  trainable params  mean runtime")
    print("-" * 75)
    for method, group in sorted(grouped.items()):
        scores = np.array([record["best_validation"]["accuracy"] for record in group])
        runtimes = np.array([record["elapsed_seconds"] for record in group])
        params = group[0]["trainable_parameters"]
        print(
            f"{method:10} {len(group):>5}  {scores.mean():.4f} ± {scores.std(ddof=0):.4f}"
            f"       {params:>12,}  {runtimes.mean() / 60:.1f} min"
        )


if __name__ == "__main__":
    main()
