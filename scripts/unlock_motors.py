#!/usr/bin/env python3
"""Unlock all motors on the SO-101 follower arm.

Disables torque on all motors so the arm can be moved freely by hand.
"""

import argparse
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def main():
    parser = argparse.ArgumentParser(description="Unlock all motors on SO-101")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="ggando_so101_follower",
        help="Robot ID for calibration lookup",
    )
    args = parser.parse_args()

    print(f"Connecting to SO-101 on {args.port}...", flush=True)

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        disable_torque_on_disconnect=True,
    )
    robot = SO101Follower(config)
    robot.connect()

    print("Disabling torque on all motors...", flush=True)
    robot.bus.disable_torque()

    print("Motors unlocked. Arm can now be moved freely.", flush=True)
    print("Press Ctrl+C to exit (torque will remain disabled).", flush=True)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting...")
        robot.disconnect()


if __name__ == "__main__":
    main()
