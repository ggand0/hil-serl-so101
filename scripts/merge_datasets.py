#!/usr/bin/env python3
"""Merge multiple LeRobotDatasets with episode filtering for HIL-SERL training."""

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def copy_episodes_from_dataset(source_dataset: LeRobotDataset, output_dataset: LeRobotDataset, task: str = ""):
    """Copy all episodes from source to output dataset."""
    # Get valid feature keys from output dataset
    valid_keys = set(output_dataset.features.keys())
    # Meta keys that shouldn't be copied
    meta_keys = {"index", "frame_index", "episode_index", "timestamp", "task_index", "task"}

    # Iterate through loaded episodes using episode_data_index
    for local_ep_idx in range(source_dataset.num_episodes):
        ep_start = source_dataset.episode_data_index["from"][local_ep_idx].item()
        ep_end = source_dataset.episode_data_index["to"][local_ep_idx].item()

        for frame_idx in range(ep_start, ep_end):
            src_frame = source_dataset[frame_idx]
            # Get timestamp from frame
            timestamp = src_frame.get("timestamp", None)
            if timestamp is not None:
                timestamp = float(timestamp)

            # Filter frame to only include valid features (exclude meta keys)
            frame = {}
            for key in src_frame:
                if key in meta_keys:
                    continue
                if key in valid_keys:
                    val = src_frame[key]
                    # Ensure scalar values have shape (1,) not ()
                    if hasattr(val, "shape") and val.shape == ():
                        val = val.unsqueeze(0) if hasattr(val, "unsqueeze") else val.reshape(1)
                    frame[key] = val

            output_dataset.add_frame(frame, task=task, timestamp=timestamp)

        output_dataset.save_episode()
        print(f"  Copied episode (frames {ep_start}-{ep_end})")


def merge_datasets(
    output_repo_id: str,
    datasets_config: list[dict],
    output_root: Path | None = None,
):
    """
    Merge multiple datasets with episode filtering.

    Args:
        output_repo_id: Repository ID for merged dataset
        datasets_config: List of dicts with keys:
            - repo_id: Source dataset repo ID
            - exclude_episodes: List of episode indices to exclude
        output_root: Output directory (default: HF cache)
    """
    output_dataset = None

    for config in datasets_config:
        repo_id = config["repo_id"]
        root = config.get("root", None)
        exclude_episodes = set(config.get("exclude_episodes", []))

        # Get total episodes from meta first
        print(f"\nLoading {repo_id}...")
        temp_dataset = LeRobotDataset(repo_id, root=root, episodes=[0], video_backend="pyav")
        total_episodes = temp_dataset.meta.total_episodes
        del temp_dataset

        # Calculate included episodes
        include_episodes = [e for e in range(total_episodes) if e not in exclude_episodes]
        print(f"  Total episodes: {total_episodes}, excluding: {exclude_episodes}")
        print(f"  Including episodes: {include_episodes}")

        # Load dataset with filtered episodes
        dataset = LeRobotDataset(repo_id, root=root, episodes=include_episodes, video_backend="pyav")
        print(f"  Loaded {dataset.num_episodes} episodes, {dataset.num_frames} frames")

        # Create output dataset on first iteration
        if output_dataset is None:
            features = dataset.meta.info["features"]
            fps = dataset.fps

            print(f"\nCreating output dataset: {output_repo_id}")
            output_dataset = LeRobotDataset.create(
                output_repo_id,
                fps=fps,
                root=output_root,
                features=features,
                use_videos=True,
                image_writer_threads=4,
                image_writer_processes=0,
            )

        # Copy episodes
        print(f"Copying episodes from {repo_id}...")
        copy_episodes_from_dataset(dataset, output_dataset)

    print(f"\nMerged dataset: {output_dataset.num_episodes} episodes, {output_dataset.num_frames} frames")
    print(f"Saved to: {output_dataset.root}")

    return output_dataset


if __name__ == "__main__":
    # Merge locked wrist v3 and v4 (local datasets)
    cache_dir = Path("/home/gota/.cache/huggingface/lerobot")
    datasets_config = [
        {
            "repo_id": "gtgando/so101_pick_lift_cube_locked_wrist_v3",
            "root": cache_dir / "gtgando" / "so101_pick_lift_cube_locked_wrist_v3",
            "exclude_episodes": [7],  # Exclude episode 7
        },
        {
            "repo_id": "gtgando/so101_pick_lift_cube_locked_wrist_v4",
            "root": cache_dir / "gtgando" / "so101_pick_lift_cube_locked_wrist_v4",
            "exclude_episodes": [],
        },
    ]

    merged = merge_datasets(
        output_repo_id="gtgando/so101_pick_lift_cube_locked_wrist_merged",
        datasets_config=datasets_config,
        output_root=cache_dir / "gtgando" / "so101_pick_lift_cube_locked_wrist_merged",
    )

    print(f"\nDone! Merged dataset at: {merged.root}")
