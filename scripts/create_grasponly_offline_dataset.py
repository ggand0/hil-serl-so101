#!/usr/bin/env python3
"""Create merged HIL-SERL offline dataset.

Merges:
- Episodes 000-014 from so101_grasp_only_v1 (positive)
- Episodes 02, 07, 08 from so101_grasp_only_negative_v1 (partial success)
"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def create_merged_offline_dataset(
    output_root: Path,
    repo_id: str = "gtgando/so101_grasp_only_offline_v1",
):
    """Create merged offline dataset for HIL-SERL."""

    cache_dir = Path("/home/gota/.cache/huggingface/lerobot/gtgando")

    positive_ds = cache_dir / "so101_grasp_only_v1"
    negative_ds = cache_dir / "so101_grasp_only_negative_v1"

    # Define which episodes to include
    episodes_to_merge = [
        # (source_dataset, source_episode_idx)
        (positive_ds, 0),
        (positive_ds, 1),
        (positive_ds, 2),
        (positive_ds, 3),
        (positive_ds, 4),
        (positive_ds, 5),
        (positive_ds, 6),
        (positive_ds, 7),
        (positive_ds, 8),
        (positive_ds, 9),
        (positive_ds, 10),
        (positive_ds, 11),
        (positive_ds, 12),
        (positive_ds, 13),
        (positive_ds, 14),
        # Negative episodes that had some success
        (negative_ds, 2),
        (negative_ds, 7),
        (negative_ds, 8),
    ]

    print(f"Merging {len(episodes_to_merge)} episodes:")
    for ds, ep in episodes_to_merge:
        print(f"  {ds.name} episode {ep}")

    # Prepare output directory
    if output_root.exists():
        print(f"\nRemoving existing output: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    # Create output structure
    data_dir = output_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    video_dir = output_root / "videos" / "chunk-000" / "observation.images.gripper_cam"
    video_dir.mkdir(parents=True)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True)

    # Process each episode
    all_frames = []
    episode_infos = []
    global_index = 0

    for new_ep_idx, (source_ds, source_ep_idx) in enumerate(episodes_to_merge):
        # Read source parquet
        pq_path = source_ds / "data" / "chunk-000" / f"episode_{source_ep_idx:06d}.parquet"
        print(f"\nProcessing: {source_ds.name}/episode_{source_ep_idx:06d}")

        if not pq_path.exists():
            raise FileNotFoundError(f"Source parquet not found: {pq_path}")

        table = pq.read_table(pq_path)
        df = table.to_pandas()
        num_frames = len(df)

        # Update indices
        df["episode_index"] = new_ep_idx
        df["index"] = range(global_index, global_index + num_frames)
        df["frame_index"] = range(num_frames)
        df["task_index"] = 0

        # Keep rewards as-is (0.0 for all in recording mode)
        # The reward classifier will provide rewards during RL training

        print(f"  Source: episode {source_ep_idx} ({num_frames} frames)")
        print(f"  Output: episode {new_ep_idx} (global index {global_index}-{global_index + num_frames - 1})")

        # Write parquet
        out_pq = data_dir / f"episode_{new_ep_idx:06d}.parquet"
        new_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(new_table, out_pq)

        # Copy video
        video_src = source_ds / "videos" / "chunk-000" / "observation.images.gripper_cam" / f"episode_{source_ep_idx:06d}.mp4"
        video_dst = video_dir / f"episode_{new_ep_idx:06d}.mp4"
        if video_src.exists():
            shutil.copy2(video_src, video_dst)
            print(f"  Video copied: {video_dst.name}")
        else:
            print(f"  Warning: video not found: {video_src}")

        # Episode info
        episode_infos.append({
            "episode_index": new_ep_idx,
            "tasks": ["grasp_only"],
            "length": num_frames,
        })

        all_frames.append(df)
        global_index += num_frames

    # Combine all frames for summary
    combined_df = pd.concat(all_frames, ignore_index=True)

    print(f"\n=== Merged Dataset ===")
    print(f"Total episodes: {len(episodes_to_merge)}")
    print(f"Total frames: {len(combined_df)}")
    print(f"Episode breakdown:")
    for i, info in enumerate(episode_infos):
        print(f"  {i}: {info['length']} frames")

    # Write episodes.jsonl
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for ep_info in episode_infos:
            f.write(json.dumps(ep_info) + "\n")

    # Write tasks.jsonl
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        f.write(json.dumps({"task_index": 0, "task": "grasp_only"}) + "\n")

    # Read source info.json and update
    with open(positive_ds / "meta" / "info.json", 'r') as f:
        info = json.load(f)

    info["total_episodes"] = len(episodes_to_merge)
    info["total_frames"] = len(combined_df)
    info["total_videos"] = len(episodes_to_merge)
    info["splits"] = {"train": f"0:{len(episodes_to_merge)}"}

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=4)

    print(f"\nOutput saved to: {output_root}")
    print(f"Repo ID: {repo_id}")

    # Verify output
    print("\n=== Verification ===")
    out_episodes = list(sorted(data_dir.glob("*.parquet")))
    out_videos = list(sorted(video_dir.glob("*.mp4")))
    print(f"Parquet files: {len(out_episodes)}")
    print(f"Video files: {len(out_videos)}")

    if len(out_episodes) != len(episodes_to_merge):
        print(f"ERROR: Expected {len(episodes_to_merge)} parquets, got {len(out_episodes)}")
    if len(out_videos) != len(episodes_to_merge):
        print(f"ERROR: Expected {len(episodes_to_merge)} videos, got {len(out_videos)}")

    # Verify indices are contiguous
    all_indices = combined_df["index"].tolist()
    expected_indices = list(range(len(combined_df)))
    if all_indices != expected_indices:
        print("ERROR: Global indices are not contiguous!")
        print(f"  Expected: 0-{len(combined_df)-1}")
        print(f"  Got: {min(all_indices)}-{max(all_indices)}")
    else:
        print("Global indices: OK (contiguous)")

    # Verify episode indices
    for new_ep_idx in range(len(episodes_to_merge)):
        ep_pq = data_dir / f"episode_{new_ep_idx:06d}.parquet"
        ep_df = pq.read_table(ep_pq).to_pandas()
        if (ep_df["episode_index"] != new_ep_idx).any():
            print(f"ERROR: Episode {new_ep_idx} has wrong episode_index values!")
        if (ep_df["frame_index"] != list(range(len(ep_df)))).any():
            print(f"ERROR: Episode {new_ep_idx} has wrong frame_index values!")

    print("Episode indices: OK")
    print("Frame indices: OK")

    return output_root


if __name__ == "__main__":
    cache_dir = Path("/home/gota/.cache/huggingface/lerobot/gtgando")
    output = cache_dir / "so101_grasp_only_offline_v1"

    create_merged_offline_dataset(
        output_root=output,
        repo_id="gtgando/so101_grasp_only_offline_v1",
    )
