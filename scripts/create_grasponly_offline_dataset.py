#!/usr/bin/env python3
"""Create merged HIL-SERL offline dataset from JSON config."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LABELS_DIR = Path("/home/gota/ggando/ml/so101-playground/data/labels")
CACHE_DIR = Path("/home/gota/.cache/huggingface/lerobot/gtgando")


def load_config(config_name: str) -> dict:
    config_path = LABELS_DIR / f"{config_name}.json"
    with open(config_path) as f:
        return json.load(f)


def create_merged_offline_dataset(config_name: str = "offline_v1"):
    config = load_config(config_name)
    repo_id = config["output_repo_id"]
    output_root = CACHE_DIR / repo_id.split("/")[-1]

    # Build episode list from config
    episodes_to_merge = []
    for source in config["sources"]:
        ds_path = Path(source["dataset_path"])
        for ep in source["episodes"]:
            episodes_to_merge.append((ds_path, ep))

    print(f"Config: {config_name}")
    print(f"Output: {output_root}")
    print(f"Merging {len(episodes_to_merge)} episodes:")
    for ds, ep in episodes_to_merge:
        print(f"  {ds.name} ep {ep}")

    # Prepare output
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    data_dir = output_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    video_dir = output_root / "videos" / "chunk-000" / "observation.images.gripper_cam"
    video_dir.mkdir(parents=True)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True)

    # Process episodes
    all_frames = []
    episode_infos = []
    global_index = 0

    for new_ep_idx, (source_ds, source_ep_idx) in enumerate(episodes_to_merge):
        pq_path = source_ds / "data" / "chunk-000" / f"episode_{source_ep_idx:06d}.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Not found: {pq_path}")

        table = pq.read_table(pq_path)
        df = table.to_pandas()
        num_frames = len(df)

        df["episode_index"] = new_ep_idx
        df["index"] = range(global_index, global_index + num_frames)
        df["frame_index"] = range(num_frames)
        df["task_index"] = 0

        out_pq = data_dir / f"episode_{new_ep_idx:06d}.parquet"
        new_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(new_table, out_pq)

        video_src = source_ds / "videos" / "chunk-000" / "observation.images.gripper_cam" / f"episode_{source_ep_idx:06d}.mp4"
        video_dst = video_dir / f"episode_{new_ep_idx:06d}.mp4"
        if video_src.exists():
            shutil.copy2(video_src, video_dst)

        episode_infos.append({
            "episode_index": new_ep_idx,
            "tasks": ["grasp_only"],
            "length": num_frames,
        })

        all_frames.append(df)
        global_index += num_frames
        print(f"  {new_ep_idx}: {source_ds.name}/ep{source_ep_idx} -> {num_frames} frames")

    combined_df = pd.concat(all_frames, ignore_index=True)

    # Write metadata
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for ep_info in episode_infos:
            f.write(json.dumps(ep_info) + "\n")

    with open(meta_dir / "tasks.jsonl", 'w') as f:
        f.write(json.dumps({"task_index": 0, "task": "grasp_only"}) + "\n")

    first_source = Path(config["sources"][0]["dataset_path"])
    with open(first_source / "meta" / "info.json") as f:
        info = json.load(f)

    info["repo_id"] = repo_id
    info["total_episodes"] = len(episodes_to_merge)
    info["total_frames"] = len(combined_df)
    info["total_videos"] = len(episodes_to_merge)
    info["splits"] = {"train": f"0:{len(episodes_to_merge)}"}

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=4)

    print(f"\nTotal: {len(episodes_to_merge)} episodes, {len(combined_df)} frames")
    print(f"Saved to: {output_root}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="offline_v1", help="Config name in data/labels/")
    args = parser.parse_args()
    create_merged_offline_dataset(args.config)
