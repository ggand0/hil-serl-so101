#!/usr/bin/env python3
"""Record video from wrist camera for debugging simulator orientation.

Usage:
    uv run python scripts/record_camera.py                    # Display live feed
    uv run python scripts/record_camera.py --record output.mp4  # Record to file
    uv run python scripts/record_camera.py --snapshot         # Save single frame
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def center_crop_square(image: np.ndarray) -> np.ndarray:
    """Center crop image to square."""
    h, w = image.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return image[y_start : y_start + size, x_start : x_start + size]


def main():
    parser = argparse.ArgumentParser(description="Record from wrist camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--record", type=str, help="Output video file path")
    parser.add_argument("--snapshot", action="store_true", help="Save single frame")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration (seconds)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    square_size = min(width, height)
    print(f"Camera: {width}x{height} @ {fps:.1f}fps -> {square_size}x{square_size} (center crop)")

    # Warm up
    for _ in range(10):
        cap.read()

    if args.snapshot:
        ret, frame = cap.read()
        if ret:
            frame = center_crop_square(frame)
            filename = f"wrist_cam_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved snapshot to {filename}")
        cap.release()
        return

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.record, fourcc, fps, (square_size, square_size))
        print(f"Recording to {args.record} for {args.duration}s...")

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Center crop to square
            frame = center_crop_square(frame)

            frame_count += 1
            elapsed = time.time() - start_time

            # Add timestamp overlay
            cv2.putText(frame, f"t={elapsed:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if writer:
                writer.write(frame)
                if elapsed >= args.duration:
                    break
            else:
                cv2.imshow("Wrist Camera (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved {frame_count} frames to {args.record}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
