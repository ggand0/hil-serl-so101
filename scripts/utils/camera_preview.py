#!/usr/bin/env python3
"""Simple camera preview with optional teleoperation."""
import argparse
import time
import cv2
import numpy as np
from pathlib import Path


def setup_teleoperation(
    follower_port: str,
    leader_port: str,
    calibration_dir: str,
    follower_id: str = "ggando_so101_follower",
    leader_id: str = "ggando_so101_leader",
):
    """Initialize leader and follower robots for teleoperation."""
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    calibration_path = Path(calibration_dir)

    # Initialize follower
    follower_config = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        calibration_dir=calibration_path / "robots" / "so101_follower",
        use_degrees=True,
    )
    follower = make_robot_from_config(follower_config)
    follower.connect()

    # Initialize leader
    leader_config = SO101LeaderConfig(
        port=leader_port,
        id=leader_id,
        calibration_dir=calibration_path / "teleoperators" / "so101_leader",
        use_degrees=True,
    )
    leader = make_teleoperator_from_config(leader_config)
    leader.connect()

    return follower, leader


def main():
    parser = argparse.ArgumentParser(description="Camera preview with optional teleop")
    parser.add_argument("--teleop", action="store_true", help="Enable teleoperation")
    parser.add_argument("--follower_port", default="/dev/ttyACM0", help="Follower port")
    parser.add_argument("--leader_port", default="/dev/ttyACM1", help="Leader port")
    parser.add_argument("--calibration_dir", default="/home/gota/.cache/huggingface/lerobot/calibration")
    parser.add_argument("--camera", default="/dev/video0", help="Camera device")
    parser.add_argument("--crop", action="store_true", help="Show training crop [0, 80, 480, 480]")
    args = parser.parse_args()

    follower = None
    leader = None

    if args.teleop:
        print("Setting up teleoperation...")
        follower, leader = setup_teleoperation(
            args.follower_port,
            args.leader_port,
            args.calibration_dir,
        )
        print("Teleop ready. Move leader arm to control follower.")

    # Camera setup
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera preview started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Teleop loop
            if args.teleop and follower and leader:
                leader_state = leader.get_action()
                follower.send_action(leader_state)

            # Display
            if args.crop:
                # Training crop: [0, 80, 480, 480]
                h, w = frame.shape[:2]
                crop_x, crop_y, crop_w, crop_h = 0, 80, 480, 480
                if crop_y + crop_h <= h and crop_x + crop_w <= w:
                    display = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                else:
                    display = frame
                cv2.imshow("Camera - Cropped (q to quit)", display)
            else:
                cv2.imshow("Camera (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if follower:
            follower.disconnect()
        if leader:
            leader.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
