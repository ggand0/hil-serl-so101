#!/usr/bin/env python3
"""Extract train/val images based on split JSON files."""
import argparse
import json
from pathlib import Path

import av


def extract_frames_from_video(video_path: Path, frame_indices: list[int], output_dir: Path, ep_idx: int):
    """Extract specific frames from a video file."""
    frame_set = set(frame_indices)
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    extracted = 0
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx in frame_set:
            img = frame.to_image()
            out_path = output_dir / f"ep{ep_idx:02d}_frame{frame_idx:03d}.jpg"
            img.save(out_path, quality=95)
            extracted += 1
            if extracted >= len(frame_indices):
                break

    container.close()
    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Dataset root path")
    parser.add_argument("--split-json", required=True, help="Split JSON file (train_episodes.json or val_episodes.json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    parser.add_argument("--camera", default="observation.images.gripper_cam", help="Camera key")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_json) as f:
        split_data = json.load(f)

    total_frames = sum(len(frames) for frames in split_data.values())
    print(f"Extracting {total_frames} frames from {len(split_data)} episodes")

    extracted_total = 0
    for ep_idx_str, frame_indices in sorted(split_data.items(), key=lambda x: int(x[0])):
        ep_idx = int(ep_idx_str)
        video_path = dataset_root / "videos" / f"chunk-000" / args.camera / f"episode_{ep_idx:06d}.mp4"

        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            continue

        extracted = extract_frames_from_video(video_path, frame_indices, output_dir, ep_idx)
        extracted_total += extracted
        print(f"Episode {ep_idx}: extracted {extracted}/{len(frame_indices)} frames")

    print(f"\nTotal extracted: {extracted_total}/{total_frames}")


if __name__ == "__main__":
    main()
