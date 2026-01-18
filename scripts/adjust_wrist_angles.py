#!/usr/bin/env python3
"""Interactive script to find correct wrist joint angles after recalibration.

Usage:
    uv run python scripts/adjust_wrist_angles.py
    uv run python scripts/adjust_wrist_angles.py --port /dev/ttyACM0
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# STS3215 resolution: 4096 steps per 360 degrees
STEPS_PER_DEG = 4096 / 360.0


def steps_to_deg(steps: int) -> float:
    return steps / STEPS_PER_DEG


def deg_to_steps(deg: float) -> int:
    return int(deg * STEPS_PER_DEG)


def main():
    parser = argparse.ArgumentParser(description="Adjust wrist angles interactively")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--flex", type=float, default=90.0, help="Initial wrist_flex angle")
    parser.add_argument("--roll", type=float, default=90.0, help="Initial wrist_roll angle")
    args = parser.parse_args()

    print("=" * 60)
    print("Wrist Angle Adjustment Tool")
    print("=" * 60)
    print()

    # Connect to robot (use RANGE_M100_100 mode and normalize=False to avoid calibration)
    print(f"Connecting to {args.port}...")
    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
        },
    )
    bus.connect()
    print("Connected!")

    # Enable torque on wrist motors so they respond to commands
    print("Enabling torque on wrist_flex and wrist_roll...")
    bus.enable_torque(["wrist_flex", "wrist_roll"])

    # Read current positions (raw steps)
    current = bus.sync_read("Present_Position", normalize=False)
    print()
    print("Current joint positions:")
    for i, name in enumerate(MOTOR_NAMES):
        steps = current[name]
        deg = steps_to_deg(steps)
        print(f"  [{i}] {name}: {deg:.1f}° (raw: {steps})")

    # Initial wrist angles (in degrees)
    wrist_flex = args.flex
    wrist_roll = args.roll

    print()
    print("Commands:")
    print("  3 <angle>  - Set wrist_flex (joint 3) angle in degrees")
    print("  4 <angle>  - Set wrist_roll (joint 4) angle in degrees")
    print("  +3 / -3    - Adjust wrist_flex by +/- 5 degrees")
    print("  +4 / -4    - Adjust wrist_roll by +/- 5 degrees")
    print("  r          - Read current positions")
    print("  q          - Quit and print config")
    print()

    # Move to initial position
    print(f"Moving to: wrist_flex={wrist_flex:.1f}°, wrist_roll={wrist_roll:.1f}°")
    bus.sync_write("Goal_Position", {
        "wrist_flex": deg_to_steps(wrist_flex),
        "wrist_roll": deg_to_steps(wrist_roll),
    }, normalize=False)
    time.sleep(0.5)

    try:
        while True:
            cmd = input(f"[flex={wrist_flex:.1f}°, roll={wrist_roll:.1f}°] > ").strip()

            if cmd == 'q' or cmd == 'quit':
                break
            elif cmd == 'r':
                current = bus.sync_read("Present_Position", normalize=False)
                print("Current positions:")
                for i, name in enumerate(MOTOR_NAMES):
                    steps = current[name]
                    deg = steps_to_deg(steps)
                    print(f"  [{i}] {name}: {deg:.1f}° (raw: {steps})")
            elif cmd == '+3':
                wrist_flex += 5
                bus.sync_write("Goal_Position", {"wrist_flex": deg_to_steps(wrist_flex)}, normalize=False)
            elif cmd == '-3':
                wrist_flex -= 5
                bus.sync_write("Goal_Position", {"wrist_flex": deg_to_steps(wrist_flex)}, normalize=False)
            elif cmd == '+4':
                wrist_roll += 5
                bus.sync_write("Goal_Position", {"wrist_roll": deg_to_steps(wrist_roll)}, normalize=False)
            elif cmd == '-4':
                wrist_roll -= 5
                bus.sync_write("Goal_Position", {"wrist_roll": deg_to_steps(wrist_roll)}, normalize=False)
            elif cmd.startswith('3 '):
                try:
                    wrist_flex = float(cmd.split()[1])
                    bus.sync_write("Goal_Position", {"wrist_flex": deg_to_steps(wrist_flex)}, normalize=False)
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd.startswith('4 '):
                try:
                    wrist_roll = float(cmd.split()[1])
                    bus.sync_write("Goal_Position", {"wrist_roll": deg_to_steps(wrist_roll)}, normalize=False)
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd:
                print("Unknown command. Use: 3 <angle>, 4 <angle>, +3, -3, +4, -4, r, q")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        print()
        print("=" * 60)
        print("Final angles:")
        print(f"  Wrist flex (joint 3): {wrist_flex:.1f}°")
        print(f"  Wrist roll (joint 4): {wrist_roll:.1f}°")
        print()
        print("Add to train_config.json robot section:")
        print(f'  "locked_joint_positions": {{"3": {wrist_flex:.1f}, "4": {wrist_roll:.1f}}}')
        print()
        print("Update fixed_reset_joint_positions:")
        print(f'  "fixed_reset_joint_positions": [0.0, -30.0, 90.0, {wrist_flex:.1f}, {wrist_roll:.1f}, 30.0]')
        print("=" * 60)
        print("Disabling torque...")
        bus.disable_torque(["wrist_flex", "wrist_roll"])
        bus.disconnect()


if __name__ == "__main__":
    main()
