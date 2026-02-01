#!/usr/bin/env python3
"""Disable motor torque on SO101 robot arms."""

import argparse
from pathlib import Path
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig


def main():
    parser = argparse.ArgumentParser(description="Unlock SO101 robot motors")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the robot",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="ggando_so101_follower",
        help="Robot ID (must match calibration)",
    )
    parser.add_argument(
        "--calibration_dir",
        type=str,
        default="/home/gota/.cache/huggingface/lerobot/calibration/robots/so101_follower",
        help="Calibration directory",
    )
    args = parser.parse_args()

    print(f"Connecting to robot on {args.port}...")

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        calibration_dir=Path(args.calibration_dir),
        disable_torque_on_disconnect=True,
    )

    robot = SO101Follower(config)
    robot.connect()

    print("Disabling torque on all motors...")
    robot.bus.sync_write("Torque_Enable", {name: False for name in robot.bus.motors})

    print("Motors unlocked. You can now move the arm freely.")
    robot.disconnect()


if __name__ == "__main__":
    main()
