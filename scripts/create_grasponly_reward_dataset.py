#!/usr/bin/env python3
"""Create labeled reward classifier dataset from JSON configs."""
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


def is_success_frame(frame_idx: int, labels: dict, total_frames: int) -> bool:
    """Check if frame is success. Frame indices are 0-based in parquet."""
    if labels is None:
        return False

    if "success_start" in labels:
        return frame_idx >= labels["success_start"]

    if "ranges" in labels:
        for start, end in labels["ranges"]:
            actual_end = end if end is not None else total_frames
            if start <= frame_idx <= actual_end:
                return True
        return False

    return False


def create_reward_dataset(config_name: str = "reward_v1"):
    config = load_config(config_name)
    repo_id = config["output_repo_id"]
    output_root = CACHE_DIR / repo_id.split("/")[-1]

    # Load all source configs
    sources = []
    for source_name in config["sources"]:
        source_config = load_config(source_name)
        sources.append(source_config)

    print(f"Config: {config_name}")
    print(f"Output: {output_root}")
    print(f"Sources: {config['sources']}")

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

    # Process all sources
    all_frames = []
    episode_infos = []
    new_ep_idx = 0
    global_index = 0

    for source in sources:
        ds_path = Path(source["dataset_path"])
        episodes = source["episodes"]

        print(f"\nSource: {ds_path.name}")

        for ep_str, labels in episodes.items():
            ep_num = int(ep_str)
            pq_path = ds_path / "data" / "chunk-000" / f"episode_{ep_num:06d}.parquet"

            if not pq_path.exists():
                print(f"  ep{ep_num}: NOT FOUND, skipping")
                continue

            table = pq.read_table(pq_path)
            df = table.to_pandas()
            num_frames = len(df)

            # Apply labels
            exclude = labels.get("exclude", []) if labels else []
            success_count = 0
            for i in range(num_frames):
                if i in exclude:
                    df.at[i, "next.reward"] = 0.0
                elif is_success_frame(i, labels, num_frames):
                    df.at[i, "next.reward"] = 1.0
                    success_count += 1
                else:
                    df.at[i, "next.reward"] = 0.0

            # Update indices
            df["episode_index"] = new_ep_idx
            df["index"] = range(global_index, global_index + num_frames)
            df["frame_index"] = range(num_frames)
            df["task_index"] = 0

            # Write parquet
            out_pq = data_dir / f"episode_{new_ep_idx:06d}.parquet"
            new_table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(new_table, out_pq)

            # Copy video
            video_src = ds_path / "videos" / "chunk-000" / "observation.images.gripper_cam" / f"episode_{ep_num:06d}.mp4"
            video_dst = video_dir / f"episode_{new_ep_idx:06d}.mp4"
            if video_src.exists():
                shutil.copy2(video_src, video_dst)

            episode_infos.append({
                "episode_index": new_ep_idx,
                "tasks": ["grasp_only"],
                "length": num_frames,
            })

            # Copy episode stats from source
            src_stats_path = ds_path / "meta" / "episodes_stats.jsonl"
            if src_stats_path.exists():
                with open(src_stats_path) as f:
                    for line in f:
                        stat = json.loads(line)
                        if stat["episode_index"] == ep_num:
                            stat["episode_index"] = new_ep_idx
                            episode_infos[-1]["stats"] = stat["stats"]
                            break

            all_frames.append(df)
            print(f"  ep{ep_num} -> {new_ep_idx}: {num_frames} frames, {success_count} success")

            new_ep_idx += 1
            global_index += num_frames

    combined_df = pd.concat(all_frames, ignore_index=True)
    total_success = (combined_df["next.reward"] == 1.0).sum()
    total_fail = (combined_df["next.reward"] == 0.0).sum()

    # Write metadata
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for ep_info in episode_infos:
            ep_data = {k: v for k, v in ep_info.items() if k != "stats"}
            f.write(json.dumps(ep_data) + "\n")

    # Write episodes_stats.jsonl
    with open(meta_dir / "episodes_stats.jsonl", 'w') as f:
        for ep_info in episode_infos:
            if "stats" in ep_info:
                stat_data = {"episode_index": ep_info["episode_index"], "stats": ep_info["stats"]}
                f.write(json.dumps(stat_data) + "\n")

    with open(meta_dir / "tasks.jsonl", 'w') as f:
        f.write(json.dumps({"task_index": 0, "task": "grasp_only"}) + "\n")

    first_source_path = Path(sources[0]["dataset_path"])
    with open(first_source_path / "meta" / "info.json") as f:
        info = json.load(f)

    info["repo_id"] = repo_id
    info["total_episodes"] = new_ep_idx
    info["total_frames"] = len(combined_df)
    info["total_videos"] = new_ep_idx
    info["splits"] = {"train": f"0:{new_ep_idx}"}

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=4)

    print(f"\n=== Summary ===")
    print(f"Episodes: {new_ep_idx}")
    print(f"Frames: {len(combined_df)}")
    print(f"Success: {total_success} ({100*total_success/len(combined_df):.1f}%)")
    print(f"Fail: {total_fail} ({100*total_fail/len(combined_df):.1f}%)")
    print(f"Saved to: {output_root}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="reward_v1", help="Config name in data/labels/")
    args = parser.parse_args()
    create_reward_dataset(args.config)
