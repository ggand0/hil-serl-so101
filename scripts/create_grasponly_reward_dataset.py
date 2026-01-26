#!/usr/bin/env python3
"""Create a labeled reward classifier dataset from annotations.

Uses the annotation file format:
  00: 34-         # frames 34+ are success
  01: 40-48, 60-  # frames 40-48 and 60+ are success
  02: all fail    # all frames are failure

Frame numbers in annotations are 1-indexed (as seen in extracted frame files).
Parquet frame_index is 0-indexed, so we subtract 1.
"""
import json
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_annotation_file(filepath: str) -> dict:
    """Parse annotation file and return dict of {dataset_path: {episode: [(start, end), ...]}}"""
    datasets = {}
    current_dataset = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # Check if it's a dataset path comment
                if '/.cache/huggingface/lerobot/' in line:
                    # Extract actual path from comment
                    path_match = re.search(r'(/home/[^\s]+)', line)
                    if path_match:
                        current_dataset = path_match.group(1)
                        datasets[current_dataset] = {}
                continue

            # Parse episode annotation: "00: 34-" or "01: 40-48, 60-"
            match = re.match(r'^(\d+):\s*(.+)$', line)
            if match and current_dataset:
                ep_num = int(match.group(1))
                annotation = match.group(2).strip()

                if annotation.lower() == 'all fail':
                    datasets[current_dataset][ep_num] = []  # Empty list = all fail
                else:
                    # Parse ranges like "34-" or "40-48, 60-"
                    ranges = []
                    for part in annotation.split(','):
                        part = part.strip()
                        range_match = re.match(r'^(\d+)-(\d+)?$', part)
                        if range_match:
                            start = int(range_match.group(1))
                            end = int(range_match.group(2)) if range_match.group(2) else None
                            ranges.append((start, end))
                    datasets[current_dataset][ep_num] = ranges

    return datasets


def is_success_frame(frame_idx_0based: int, ranges: list, total_frames: int) -> bool:
    """Check if frame is in success range. frame_idx is 0-based, ranges are 1-based."""
    if not ranges:
        return False

    frame_1based = frame_idx_0based + 1  # Convert to 1-indexed for comparison

    for start, end in ranges:
        actual_end = end if end is not None else total_frames
        if start <= frame_1based <= actual_end:
            return True
    return False


def create_reward_classifier_dataset(
    annotation_file: str,
    output_root: Path,
    repo_id: str = "gtgando/so101_grasp_only_reward_v1",
):
    """Create a reward classifier dataset combining both positive and negative samples."""

    # Parse annotations
    datasets = parse_annotation_file(annotation_file)
    print(f"Parsed {len(datasets)} datasets from annotations")
    for ds_path, episodes in datasets.items():
        print(f"  {ds_path}: {len(episodes)} episodes")

    # Prepare output directory
    if output_root.exists():
        print(f"Removing existing output: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    # Collect all data
    all_data = []
    all_videos = []
    episode_infos = []
    episode_counter = 0
    global_index = 0

    for ds_path, annotations in datasets.items():
        ds_root = Path(ds_path)
        print(f"\nProcessing: {ds_root}")

        # Read episodes.jsonl to get frame counts
        episodes_file = ds_root / "meta" / "episodes.jsonl"
        episode_lengths = {}
        with open(episodes_file, 'r') as f:
            for line in f:
                ep_info = json.loads(line)
                episode_lengths[ep_info["episode_index"]] = ep_info["length"]

        for ep_num, ranges in annotations.items():
            # Read parquet for this episode
            pq_path = ds_root / "data" / "chunk-000" / f"episode_{ep_num:06d}.parquet"
            if not pq_path.exists():
                print(f"  Warning: {pq_path} not found, skipping")
                continue

            table = pq.read_table(pq_path)
            df = table.to_pandas()
            total_frames = len(df)

            # Label frames based on annotations
            success_count = 0
            for i in range(total_frames):
                is_success = is_success_frame(i, ranges, total_frames)
                df.at[i, "next.reward"] = 1.0 if is_success else 0.0
                if is_success:
                    success_count += 1

            # Update episode index and global index
            df["episode_index"] = episode_counter
            df["index"] = range(global_index, global_index + total_frames)

            # Reset frame_index to 0-based within episode
            df["frame_index"] = range(total_frames)

            all_data.append(df)

            # Track video path
            video_src = ds_root / "videos" / "chunk-000" / "observation.images.gripper_cam" / f"episode_{ep_num:06d}.mp4"
            all_videos.append((video_src, episode_counter))

            # Episode info
            episode_infos.append({
                "episode_index": episode_counter,
                "tasks": ["grasp_only"],
                "length": total_frames,
            })

            print(f"  Episode {ep_num} -> {episode_counter}: {total_frames} frames, {success_count} success ({100*success_count/total_frames:.1f}%)")

            episode_counter += 1
            global_index += total_frames

    # Combine all data
    import pandas as pd
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"\n=== Combined Dataset ===")
    print(f"Total episodes: {episode_counter}")
    print(f"Total frames: {len(combined_df)}")
    success_total = (combined_df["next.reward"] == 1.0).sum()
    print(f"Success frames: {success_total} ({100*success_total/len(combined_df):.1f}%)")

    # Create output structure
    data_dir = output_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    video_dir = output_root / "videos" / "chunk-000" / "observation.images.gripper_cam"
    video_dir.mkdir(parents=True)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True)

    # Write parquet files per episode
    for ep_idx in range(episode_counter):
        ep_df = combined_df[combined_df["episode_index"] == ep_idx].copy()
        out_pq = data_dir / f"episode_{ep_idx:06d}.parquet"
        table = pa.Table.from_pandas(ep_df, preserve_index=False)
        pq.write_table(table, out_pq)

    # Copy videos
    print("\nCopying videos...")
    for video_src, ep_idx in all_videos:
        video_dst = video_dir / f"episode_{ep_idx:06d}.mp4"
        shutil.copy2(video_src, video_dst)

    # Write episodes.jsonl
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for ep_info in episode_infos:
            f.write(json.dumps(ep_info) + "\n")

    # Write tasks.jsonl
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        f.write(json.dumps({"task_index": 0, "task": "grasp_only"}) + "\n")

    # Read source info.json and update
    source_info_path = list(datasets.keys())[0]
    with open(Path(source_info_path) / "meta" / "info.json", 'r') as f:
        info = json.load(f)

    info["total_episodes"] = episode_counter
    info["total_frames"] = len(combined_df)
    info["total_videos"] = episode_counter
    info["splits"] = {"train": f"0:{episode_counter}"}

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=4)

    print(f"\nOutput saved to: {output_root}")
    print(f"Repo ID: {repo_id}")

    return output_root


if __name__ == "__main__":
    cache_dir = Path("/home/gota/.cache/huggingface/lerobot/gtgando")

    annotation_file = "/home/gota/ggando/ml/so101-playground/data/reward_classifier_grasponly_v1.txt"
    output = cache_dir / "so101_grasp_only_reward_v1"

    create_reward_classifier_dataset(
        annotation_file=annotation_file,
        output_root=output,
        repo_id="gtgando/so101_grasp_only_reward_v1",
    )
