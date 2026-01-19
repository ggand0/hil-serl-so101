#!/usr/bin/env python3
"""Record video from wrist camera for debugging simulator orientation.

Usage:
    uv run python scripts/record_camera.py                    # Display live feed
    uv run python scripts/record_camera.py --record output.mp4  # Record to file
    uv run python scripts/record_camera.py --snapshot         # Save single frame
    uv run python scripts/record_camera.py --record output.mp4 --random_gripper  # With random gripper motion
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
    parser.add_argument("--preview", action="store_true", help="Show preview window while recording")
    parser.add_argument("--random_gripper", action="store_true", help="Randomly open/close gripper during recording")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--gripper_interval", type=float, default=1.0, help="Average interval between gripper changes (seconds)")
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

    # Initialize robot for random gripper control
    robot = None
    lerobot_robot = None
    if args.random_gripper:
        from src.deploy.robot import SO101Robot
        robot = SO101Robot(port=args.robot_port)
        if not robot.connect():
            print("Failed to connect to robot. Continuing without gripper control.")
            robot = None
        else:
            print(f"Robot connected for random gripper control (interval ~{args.gripper_interval}s)")
            # Disable torque on arm motors to allow manual movement
            lerobot_robot = robot._robot
            arm_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            lerobot_robot.bus.disable_torque(arm_motors)
            print("Arm motors torque disabled - you can manually move the arm")
            print("Gripper will randomly open/close during recording")

    start_time = time.time()
    frame_count = 0
    gripper_state = 1.0  # Start open
    next_gripper_change = start_time + random.uniform(0.5, args.gripper_interval * 2)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Center crop to square
            frame = center_crop_square(frame)

            frame_count += 1
            elapsed = time.time() - start_time
            current_time = time.time()

            # Random gripper control
            if robot is not None and current_time >= next_gripper_change:
                # Toggle gripper state with some randomness
                if gripper_state > 0:
                    gripper_state = random.uniform(-1.0, -0.5)  # Close
                else:
                    gripper_state = random.uniform(0.5, 1.0)  # Open

                # Send gripper command only (arm is free to move manually)
                robot.send_gripper(gripper_state)

                # Schedule next change
                next_gripper_change = current_time + random.uniform(
                    args.gripper_interval * 0.5,
                    args.gripper_interval * 1.5
                )

            if writer:
                writer.write(frame)
                if args.preview:
                    # Add timestamp for preview only (not saved)
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"t={elapsed:.2f}s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Recording (q to quit)", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if elapsed >= args.duration:
                    break
            else:
                # Add timestamp for live preview
                cv2.putText(frame, f"t={elapsed:.2f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
        if robot is not None:
            robot.disconnect()
            print("Robot disconnected")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
