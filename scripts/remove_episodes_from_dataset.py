#!/usr/bin/env python3
"""Remove specific episodes from a LeRobot dataset and reindex."""

import argparse
import json
import shutil
from pathlib import Path
import pandas as pd


def remove_episodes(dataset_path: str, episodes_to_remove: list[int], dry_run: bool = True):
    """Remove episodes from dataset and reindex remaining episodes."""
    dataset_path = Path(dataset_path)

    # Load metadata
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    print(f"Dataset has {total_episodes} episodes")
    print(f"Removing episodes: {episodes_to_remove}")

    # Validate episodes exist
    for ep in episodes_to_remove:
        if ep >= total_episodes:
            raise ValueError(f"Episode {ep} doesn't exist (max: {total_episodes - 1})")

    # Build mapping: old_index -> new_index (None if removed)
    new_indices = {}
    new_idx = 0
    for old_idx in range(total_episodes):
        if old_idx in episodes_to_remove:
            new_indices[old_idx] = None
        else:
            new_indices[old_idx] = new_idx
            new_idx += 1

    new_total = new_idx
    print(f"New total episodes: {new_total}")

    if dry_run:
        print("\n[DRY RUN] Would perform the following:")

    # Remove and rename parquet files
    data_dir = dataset_path / "data" / "chunk-000"
    print(f"\nProcessing parquet files in {data_dir}...")

    for old_idx in range(total_episodes):
        old_path = data_dir / f"episode_{old_idx:06d}.parquet"
        new_idx = new_indices[old_idx]

        if new_idx is None:
            print(f"  DELETE: {old_path.name}")
            if not dry_run:
                old_path.unlink()
        elif old_idx != new_idx:
            new_path = data_dir / f"episode_{new_idx:06d}.parquet"
            print(f"  RENAME: {old_path.name} -> {new_path.name}")
            if not dry_run:
                # Also update episode_index in parquet
                df = pd.read_parquet(old_path)
                df["episode_index"] = new_idx
                # Recalculate index (global frame index)
                # This needs to be done after all renaming
                df.to_parquet(old_path)
                old_path.rename(new_path)

    # Remove and rename video files
    video_dir = dataset_path / "videos" / "chunk-000" / "observation.images.gripper_cam"
    print(f"\nProcessing video files in {video_dir}...")

    for old_idx in range(total_episodes):
        old_path = video_dir / f"episode_{old_idx:06d}.mp4"
        new_idx = new_indices[old_idx]

        if new_idx is None:
            print(f"  DELETE: {old_path.name}")
            if not dry_run:
                old_path.unlink()
        elif old_idx != new_idx:
            new_path = video_dir / f"episode_{new_idx:06d}.mp4"
            print(f"  RENAME: {old_path.name} -> {new_path.name}")
            if not dry_run:
                old_path.rename(new_path)

    # Update episodes.jsonl
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        print(f"\nUpdating {episodes_path}...")
        with open(episodes_path) as f:
            episodes = [json.loads(line) for line in f]

        new_episodes = []
        for ep in episodes:
            old_idx = ep["episode_index"]
            new_idx = new_indices.get(old_idx)
            if new_idx is not None:
                ep["episode_index"] = new_idx
                new_episodes.append(ep)

        if not dry_run:
            with open(episodes_path, "w") as f:
                for ep in new_episodes:
                    f.write(json.dumps(ep) + "\n")

    # Update info.json
    print(f"\nUpdating {info_path}...")
    if not dry_run:
        # Recalculate total frames
        total_frames = 0
        for idx in range(new_total):
            parquet_path = data_dir / f"episode_{idx:06d}.parquet"
            df = pd.read_parquet(parquet_path)
            total_frames += len(df)

        info["total_episodes"] = new_total
        info["total_videos"] = new_total
        info["total_frames"] = total_frames
        info["splits"]["train"] = f"0:{new_total}"

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    # Recalculate global indices in parquet files
    if not dry_run:
        print("\nRecalculating global indices...")
        global_idx = 0
        for ep_idx in range(new_total):
            parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(parquet_path)
            df["index"] = range(global_idx, global_idx + len(df))
            df.to_parquet(parquet_path)
            global_idx += len(df)

    print("\nDone!" if not dry_run else "\n[DRY RUN] No changes made. Run with --execute to apply.")


def main():
    parser = argparse.ArgumentParser(description="Remove episodes from LeRobot dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--remove", type=int, nargs="+", required=True, help="Episode indices to remove")
    parser.add_argument("--execute", action="store_true", help="Actually perform the removal (default: dry run)")
    args = parser.parse_args()

    remove_episodes(args.dataset, args.remove, dry_run=not args.execute)


if __name__ == "__main__":
    main()
