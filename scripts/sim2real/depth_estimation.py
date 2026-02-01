#!/usr/bin/env python3
"""Depth estimation using Depth Anything V2 Small.

Runs monocular depth estimation on camera feed or image input.

Setup (first time only):
    # Clone Depth Anything V2 and download model
    uv run python scripts/depth_estimation.py --setup

Usage:
    # Live camera feed with depth visualization
    uv run python scripts/depth_estimation.py --camera 0

    # Single image inference
    uv run python scripts/depth_estimation.py --image path/to/image.jpg

    # Save depth output
    uv run python scripts/depth_estimation.py --camera 0 --output depth_video.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Depth Anything V2 paths
DEPTH_ANYTHING_DIR = project_root / "third_party" / "Depth-Anything-V2"
CHECKPOINT_DIR = project_root / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "depth_anything_v2_vits.pth"


def setup_depth_anything():
    """Clone Depth Anything V2 and download checkpoint."""
    import subprocess

    # Create directories
    third_party_dir = project_root / "third_party"
    third_party_dir.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Clone repo if not exists
    if not DEPTH_ANYTHING_DIR.exists():
        print("Cloning Depth Anything V2...")
        subprocess.run([
            "git", "clone",
            "https://github.com/DepthAnything/Depth-Anything-V2.git",
            str(DEPTH_ANYTHING_DIR)
        ], check=True)
        print("Done.")
    else:
        print(f"Depth Anything V2 already exists at {DEPTH_ANYTHING_DIR}")

    # Download checkpoint if not exists
    if not CHECKPOINT_PATH.exists():
        print("Downloading Depth Anything V2 Small checkpoint...")
        url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
        subprocess.run([
            "wget", "-O", str(CHECKPOINT_PATH), url
        ], check=True)
        print(f"Saved to {CHECKPOINT_PATH}")
    else:
        print(f"Checkpoint already exists at {CHECKPOINT_PATH}")

    print("\nSetup complete! You can now run depth estimation.")


def load_model(device="cuda"):
    """Load Depth Anything V2 Small model."""
    import torch

    # Add Depth Anything V2 to path
    sys.path.insert(0, str(DEPTH_ANYTHING_DIR))

    from depth_anything_v2.dpt import DepthAnythingV2

    # Model config for Small (vits)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }

    model = DepthAnythingV2(**model_configs['vits'])
    model.load_state_dict(torch.load(str(CHECKPOINT_PATH), map_location='cpu'))
    model = model.to(device).eval()

    return model


def depth_to_colormap(depth: np.ndarray, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Convert depth map to colored visualization."""
    # Normalize to 0-255
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(depth_uint8, colormap)

    return colored


def center_crop_square(image: np.ndarray) -> np.ndarray:
    """Center crop to square."""
    h, w = image.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return image[y_start:y_start + size, x_start:x_start + size]


def run_camera_inference(model, camera_index: int, output_path: str = None,
                         resolution: int = 480, show_fps: bool = True):
    """Run depth estimation on live camera feed."""
    import torch

    device = next(model.parameters()).device

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        return 1

    # Warm up
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {w}x{h}")

    # Video writer for combined output
    writer = None
    if output_path:
        # Ensure video extension for camera mode
        if not output_path.endswith(('.mp4', '.avi', '.mkv')):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
            print(f"Note: Camera mode requires video format, using: {output_path}")
        frame_size = (resolution * 2, resolution)  # Side by side
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 15.0, frame_size)
        print(f"Recording to: {output_path}")

    print("Press 'q' to quit")

    frame_count = 0
    start_time = time.time()
    fps_update_interval = 10
    current_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Preprocess
            frame_cropped = center_crop_square(frame)
            frame_resized = cv2.resize(frame_cropped, (resolution, resolution))

            # Run inference
            with torch.no_grad():
                depth = model.infer_image(frame_resized)

            # Visualize
            depth_colored = depth_to_colormap(depth)
            depth_resized = cv2.resize(depth_colored, (resolution, resolution))

            # Side by side display
            combined = np.hstack([frame_resized, depth_resized])

            # Add FPS overlay
            frame_count += 1
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed

            if show_fps:
                cv2.putText(combined, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display
            cv2.imshow("RGB | Depth (q to quit)", combined)

            # Save
            if writer:
                writer.write(combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved to {output_path}")
        cv2.destroyAllWindows()

    return 0


def run_image_inference(model, image_path: str, output_path: str = None):
    """Run depth estimation on a single image."""
    import torch

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image {image_path}")
        return 1

    print(f"Input image: {image.shape[1]}x{image.shape[0]}")

    # Run inference
    with torch.no_grad():
        depth = model.infer_image(image)

    print(f"Depth map: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Visualize
    depth_colored = depth_to_colormap(depth)

    # Resize depth to match input
    depth_colored = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))

    # Side by side
    combined = np.hstack([image, depth_colored])

    # Save or display
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"Saved to {output_path}")
    else:
        cv2.imshow("RGB | Depth (press any key)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 Small inference")
    parser.add_argument("--setup", action="store_true", help="Setup Depth Anything V2 (clone repo + download model)")
    parser.add_argument("--camera", type=int, default=None, help="Camera index for live inference")
    parser.add_argument("--image", type=str, default=None, help="Input image path")
    parser.add_argument("--output", type=str, default=None, help="Output path (video for camera, image for single)")
    parser.add_argument("--resolution", type=int, default=480, help="Processing resolution (default: 480)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu, default: cuda)")
    args = parser.parse_args()

    # Setup mode
    if args.setup:
        setup_depth_anything()
        return 0

    # Check setup
    if not DEPTH_ANYTHING_DIR.exists() or not CHECKPOINT_PATH.exists():
        print("ERROR: Depth Anything V2 not set up. Run with --setup first:")
        print("  uv run python scripts/depth_estimation.py --setup")
        return 1

    # Need either camera or image
    if args.camera is None and args.image is None:
        print("ERROR: Specify --camera or --image")
        parser.print_help()
        return 1

    # Check device
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Loading model on {args.device}...")
    model = load_model(device=args.device)
    print("Model loaded.")

    # Run inference
    if args.camera is not None:
        return run_camera_inference(model, args.camera, args.output, args.resolution)
    else:
        return run_image_inference(model, args.image, args.output)


if __name__ == "__main__":
    sys.exit(main())
