#!/usr/bin/env python3
"""Extract frames from recorded episodes for reward classifier labeling."""

import argparse
import subprocess
from pathlib import Path

def extract_frames(
    dataset_path: str,
    output_dir: str,
    skip_episodes: list[int],
    crop_params: tuple[int, int, int, int] = (0, 80, 480, 480),  # top, left, height, width
    sample_rate: int = 3,  # Extract every Nth frame
):
    """Extract and crop frames from video episodes using ffmpeg."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = dataset_path / "videos" / "chunk-000" / "observation.images.gripper_cam"

    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return

    episodes = sorted(video_dir.glob("episode_*.mp4"))
    print(f"Found {len(episodes)} episodes")
    print(f"Skipping episodes: {skip_episodes}")

    top, left, height, width = crop_params
    # ffmpeg crop filter: crop=w:h:x:y
    crop_filter = f"crop={width}:{height}:{left}:{top}"

    for ep_path in episodes:
        ep_num = int(ep_path.stem.split("_")[1])

        if ep_num in skip_episodes:
            print(f"Skipping episode {ep_num}")
            continue

        print(f"Processing episode {ep_num}...")

        ep_output_dir = output_dir / f"episode_{ep_num:03d}"
        ep_output_dir.mkdir(exist_ok=True)

        # Use ffmpeg to extract frames with crop
        # -vf "select=not(mod(n\,3)),crop=..." extracts every 3rd frame and crops
        cmd = [
            "ffmpeg", "-y", "-i", str(ep_path),
            "-vf", f"select=not(mod(n\\,{sample_rate})),{crop_filter}",
            "-vsync", "vfr",
            str(ep_output_dir / "frame_%04d.jpg")
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error: {result.stderr[:200]}")
        else:
            frame_count = len(list(ep_output_dir.glob("frame_*.jpg")))
            print(f"  Saved {frame_count} frames")

    print(f"\nFrames saved to: {output_dir}")
    print("Review frames and identify success frames (cube grasped and lifted)")


def main():
    parser = argparse.ArgumentParser(description="Extract frames for reward classifier labeling")
    parser.add_argument("--dataset", default="/home/gota/.cache/huggingface/lerobot/gtgando/so101_grasp_only_v3_lamp")
    parser.add_argument("--output", default="/tmp/frames_for_labeling")
    parser.add_argument("--skip", type=int, nargs="+", default=[19, 23], help="Episodes to skip")
    parser.add_argument("--sample-rate", type=int, default=3, help="Extract every Nth frame")
    parser.add_argument("--crop", type=int, nargs=4, default=[0, 80, 480, 480],
                        help="Crop params: top left height width")
    args = parser.parse_args()

    extract_frames(
        args.dataset,
        args.output,
        args.skip,
        tuple(args.crop),
        args.sample_rate,
    )


if __name__ == "__main__":
    main()
