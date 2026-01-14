#!/usr/bin/env python3
"""Capture wrist camera images at different gripper positions.

Exports images for closed, half-open, and fully-open gripper states.

Usage:
    uv run python scripts/capture_gripper_states.py
    uv run python scripts/capture_gripper_states.py --output_dir ./gripper_images
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deploy.robot import SO101Robot
from src.deploy.controllers import IKController


# Gripper positions (policy values: -1=closed, 1=open)
GRIPPER_STATES = {
    "closed": -1.0,      # 0% physical
    "half_open": 0.0,    # 50% physical
    "fully_open": 1.0,   # 100% physical
}

# Joint offset correction
ELBOW_FLEX_OFFSET_RAD = np.deg2rad(-12.5)


def apply_joint_offset(joints: np.ndarray) -> np.ndarray:
    corrected = joints.copy()
    corrected[2] += ELBOW_FLEX_OFFSET_RAD
    return corrected


def center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return image[y_start:y_start + size, x_start:x_start + size]


def main():
    parser = argparse.ArgumentParser(description="Capture gripper state images")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for images")
    parser.add_argument("--settle_time", type=float, default=2.0, help="Time to wait after gripper move")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gripper State Image Capture")
    print("=" * 60)

    # Initialize robot
    print("\n[1/3] Initializing robot...")
    robot = SO101Robot(port=args.robot_port)
    if not robot.connect():
        print("Failed to connect to robot. Exiting.")
        return

    # Initialize IK controller
    print("\n[2/3] Initializing IK controller...")
    try:
        ik = IKController()
    except Exception as e:
        print(f"Failed to initialize IK: {e}")
        robot.disconnect()
        return

    # Initialize camera
    print("\n[3/3] Initializing camera...")
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Failed to open camera. Exiting.")
        robot.disconnect()
        return

    # Warm up camera
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened: {w}x{h}")

    # Move to a neutral position first
    print("\nMoving to capture position...")
    current_joints = robot.get_joint_positions_radians()
    # Keep current arm position, just ensure we're ready
    time.sleep(0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        for state_name, gripper_value in GRIPPER_STATES.items():
            print(f"\n[{state_name}] Setting gripper to {gripper_value} ({state_name})...")

            # Move gripper
            current_joints = robot.get_joint_positions_radians()
            robot.send_action(current_joints, gripper_value)

            # Wait for gripper to settle
            time.sleep(args.settle_time)

            # Read back actual gripper position
            actual_gripper = robot.get_gripper_position()
            print(f"  Commanded: {gripper_value}, Actual: {actual_gripper:.2f}")

            # Flush camera buffer (discard stale frames)
            for _ in range(10):
                cap.read()

            # Capture fresh frame
            ret, frame = cap.read()
            if not ret:
                print(f"  ERROR: Failed to capture frame for {state_name}")
                continue

            # Process and save
            cropped = center_crop_square(frame)
            resized = cv2.resize(cropped, (480, 480))

            filename = f"gripper_{state_name}_{timestamp}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), resized)
            print(f"  Saved: {filepath}")

        print("\n" + "=" * 60)
        print("Capture complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Return gripper to neutral
        print("\nReturning gripper to half-open...")
        current_joints = robot.get_joint_positions_radians()
        robot.send_action(current_joints, 0.0)
        time.sleep(0.5)

        cap.release()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
