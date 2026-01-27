#!/usr/bin/env python3
"""Generate episodes_stats.jsonl for a dataset."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def compute_stats(values):
    """Compute min, max, mean, std for array."""
    arr = np.array(values)
    return {
        "min": arr.min(axis=0).tolist() if arr.ndim > 1 else [float(arr.min())],
        "max": arr.max(axis=0).tolist() if arr.ndim > 1 else [float(arr.max())],
        "mean": arr.mean(axis=0).tolist() if arr.ndim > 1 else [float(arr.mean())],
        "std": arr.std(axis=0).tolist() if arr.ndim > 1 else [float(arr.std())],
        "count": [len(arr)],
    }


def generate_episodes_stats(dataset_root: Path):
    """Generate episodes_stats.jsonl for dataset."""
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"

    pq_files = sorted(data_dir.glob("*.parquet"))
    print(f"Processing {len(pq_files)} episodes...")

    stats_lines = []

    for pq_file in pq_files:
        table = pq.read_table(pq_file)
        df = table.to_pandas()

        ep_idx = df["episode_index"].iloc[0]
        ep_stats = {"episode_index": int(ep_idx), "stats": {}}

        # Compute stats for each column
        for col in df.columns:
            if col in ["index", "episode_index", "frame_index", "task_index"]:
                continue

            values = df[col].tolist()

            # Skip if empty or all None
            if not values or all(v is None for v in values):
                continue

            try:
                ep_stats["stats"][col] = compute_stats(values)
            except Exception as e:
                print(f"  Warning: Could not compute stats for {col}: {e}")

        stats_lines.append(json.dumps(ep_stats))
        print(f"  Episode {ep_idx}: {len(df)} frames")

    # Write to file
    stats_file = meta_dir / "episodes_stats.jsonl"
    with open(stats_file, "w") as f:
        f.write("\n".join(stats_lines) + "\n")

    print(f"\nSaved to: {stats_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path to dataset root")
    args = parser.parse_args()

    generate_episodes_stats(Path(args.dataset_root))
