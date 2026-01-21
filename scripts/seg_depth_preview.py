#!/usr/bin/env python3
"""Live seg+depth preview from camera.

Shows RGB | Segmentation | Depth side by side with color legend.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root BEFORE any other imports
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

# Import perception models (must be after sys.path.insert, before pick-101 paths added)
from src.deploy.perception import SegmentationModel, DepthModel


def parse_args():
    parser = argparse.ArgumentParser(description="Live seg+depth preview from camera")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--seg_checkpoint",
        type=str,
        default="/home/gota/ggando/ml/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt",
        help="Path to EfficientViT segmentation checkpoint",
    )
    parser.add_argument(
        "--preview_size",
        type=int,
        default=320,
        help="Size of each preview panel (default: 320)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save preview video to file (optional)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target FPS for preview (default: 10)",
    )
    parser.add_argument(
        "--pick101_root",
        type=str,
        default="/home/gota/ggando/ml/pick-101",
        help="Path to pick-101 repository",
    )
    parser.add_argument(
        "--debug_seg",
        action="store_true",
        help="Enable debug logging for segmentation model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Add pick-101 paths
    pick101_root = Path(args.pick101_root)
    sys.path.insert(0, str(pick101_root))
    sys.path.insert(0, str(pick101_root / "external" / "robobase"))

    print("=" * 60)
    print("Seg+Depth Live Preview")
    print("=" * 60)

    # Load models
    print("\n[1/3] Loading segmentation model...")
    seg_model = SegmentationModel(args.seg_checkpoint, device=args.device, debug=args.debug_seg)
    if not seg_model.load():
        print("Failed to load segmentation model")
        return 1

    print("\n[2/3] Loading depth model...")
    depth_model = DepthModel(device=args.device)
    if not depth_model.load():
        print("Failed to load depth model")
        return 1

    print(f"\n[3/3] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return 1

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Segmentation color map (BGR for OpenCV)
    seg_colors = np.array([
        [0, 0, 0],        # 0: unlabeled - black
        [128, 128, 128],  # 1: background - gray
        [0, 128, 0],      # 2: table - green
        [0, 165, 255],    # 3: cube - orange
        [255, 0, 0],      # 4: static_finger - blue
        [255, 0, 255],    # 5: moving_finger - magenta
    ], dtype=np.uint8)
    class_names = ["unlbl", "bg", "table", "cube", "s_fgr", "m_fgr"]

    # Video writer
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.output, fourcc, args.fps, (args.preview_size * 3, args.preview_size)
        )
        print(f"Recording to: {args.output}")

    print("\n" + "=" * 60)
    print("Press 'q' to quit, 's' to save current frame")
    print("=" * 60 + "\n")

    frame_count = 0
    start_time = time.time()
    target_dt = 1.0 / args.fps

    try:
        while True:
            loop_start = time.time()

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            # Center crop to square
            h, w = frame.shape[:2]
            size = min(h, w)
            y_start = (h - size) // 2
            x_start = (w - size) // 2
            frame_cropped = frame[y_start:y_start + size, x_start:x_start + size]

            # Run inference
            seg_mask = seg_model.predict(frame_cropped)
            disparity = depth_model.predict(frame_cropped)

            # Create preview panels
            preview_size = args.preview_size

            # RGB panel
            rgb_panel = cv2.resize(frame_cropped, (preview_size, preview_size))

            # Segmentation panel (colorized)
            seg_colored = seg_colors[seg_mask]
            seg_panel = cv2.resize(seg_colored, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST)

            # Add color legend to seg panel
            legend_y = preview_size - 15
            legend_x = 5
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, (name, color) in enumerate(zip(class_names, seg_colors)):
                x = legend_x + i * 52
                # Colored square
                cv2.rectangle(seg_panel, (x, legend_y - 10), (x + 10, legend_y), tuple(int(c) for c in color), -1)
                cv2.rectangle(seg_panel, (x, legend_y - 10), (x + 10, legend_y), (255, 255, 255), 1)
                # Label
                cv2.putText(seg_panel, name, (x + 12, legend_y - 1), font, 0.35, (255, 255, 255), 1)

            # Depth panel (colormap)
            depth_colored = cv2.applyColorMap(disparity, cv2.COLORMAP_INFERNO)
            depth_panel = cv2.resize(depth_colored, (preview_size, preview_size))

            # Add labels to panels
            label_y = 20
            cv2.putText(rgb_panel, "RGB", (10, label_y), font, 0.6, (255, 255, 255), 2)
            cv2.putText(seg_panel, "Segmentation", (10, label_y), font, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_panel, "Depth", (10, label_y), font, 0.6, (255, 255, 255), 2)

            # Combine panels
            preview = np.hstack([rgb_panel, seg_panel, depth_panel])

            # Add FPS counter
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(preview, f"FPS: {fps:.1f}", (preview.shape[1] - 80, 20), font, 0.5, (0, 255, 0), 1)

            # Show preview
            cv2.imshow("Seg+Depth Preview (q=quit, s=save)", preview)

            # Write to video
            if video_writer is not None:
                video_writer.write(preview)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"seg_depth_preview_{timestamp}.png", preview)
                cv2.imwrite(f"rgb_{timestamp}.png", frame_cropped)
                cv2.imwrite(f"seg_{timestamp}.png", seg_colored)
                cv2.imwrite(f"depth_{timestamp}.png", depth_colored)
                print(f"Saved frames with timestamp {timestamp}")

            # Rate limiting
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < target_dt:
                time.sleep(target_dt - loop_elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved video: {args.output}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
