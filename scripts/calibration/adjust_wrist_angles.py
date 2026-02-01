#!/usr/bin/env python3
"""Interactive script to find correct wrist joint angles after recalibration.

Uses IK reset motion to safely lift arm before adjusting wrist angles.

Usage:
    uv run python scripts/adjust_wrist_angles.py
"""

import argparse
import time
from pathlib import Path

import numpy as np

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deploy.robot import SO101Robot
from src.deploy.controllers import IKController


SAFE_JOINTS = np.zeros(5)  # Extended forward - safe for IK movements


def ik_reset(robot, ik):
    """IK reset motion matching ik_grasp_demo.py pattern."""
    # Step 0: Move to reset position (all zeros)
    print("Step 0: Moving to reset position...")
    robot.send_action(SAFE_JOINTS, 1.0)
    time.sleep(3.0)

    # Set wrist to top-down orientation (both π/2)
    print("Setting top-down wrist orientation...")
    joints = robot.get_joint_positions_radians()
    joints[3] = np.pi / 2
    joints[4] = np.pi / 2
    robot.send_action(joints, 1.0)
    time.sleep(1.0)

    # Sync IK and show position
    ik.sync_joint_positions(robot.get_joint_positions_radians())
    ee = ik.get_ee_position()
    print(f"EE position: {ee}")


def main():
    parser = argparse.ArgumentParser(description="Adjust wrist angles interactively with IK reset")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--flex", type=float, default=90.0, help="Initial wrist_flex angle (degrees)")
    parser.add_argument("--roll", type=float, default=90.0, help="Initial wrist_roll angle (degrees)")
    args = parser.parse_args()

    print("=" * 60)
    print("Wrist Angle Adjustment Tool (with IK Reset)")
    print("=" * 60)

    # Connect robot
    print(f"\nConnecting to {args.port}...")
    robot = SO101Robot(port=args.port)
    if not robot.connect():
        print("Failed to connect")
        return

    # Initialize IK
    print("Initializing IK controller...")
    ik = IKController()
    print("Ready!")

    # Show current positions
    joints_rad = robot.get_joint_positions_radians()
    joints_deg = np.rad2deg(joints_rad)
    print(f"\nCurrent joints (deg): {joints_deg}")

    # Lift to safe height
    ik_reset(robot, ik)

    # Initial wrist angles
    wrist_flex = args.flex
    wrist_roll = args.roll

    print()
    print("Commands:")
    print("  3 <angle>  - Set wrist_flex (joint 3) angle in degrees")
    print("  4 <angle>  - Set wrist_roll (joint 4) angle in degrees")
    print("  +3 / -3    - Adjust wrist_flex by +/- 5 degrees")
    print("  +4 / -4    - Adjust wrist_roll by +/- 5 degrees")
    print("  r          - Read current positions")
    print("  l          - Lift to safe height again")
    print("  q          - Quit and print config")
    print()

    # Move to initial wrist position
    print(f"Moving to: wrist_flex={wrist_flex:.1f}°, wrist_roll={wrist_roll:.1f}°")
    joints = robot.get_joint_positions_radians()
    joints[3] = np.deg2rad(wrist_flex)
    joints[4] = np.deg2rad(wrist_roll)
    robot.send_action(joints, 1.0)
    time.sleep(0.5)

    try:
        while True:
            cmd = input(f"[flex={wrist_flex:.1f}°, roll={wrist_roll:.1f}°] > ").strip()

            if cmd == 'q' or cmd == 'quit':
                break
            elif cmd == 'r':
                joints_rad = robot.get_joint_positions_radians()
                joints_deg = np.rad2deg(joints_rad)
                print(f"Current joints (deg): {joints_deg}")
                ik.sync_joint_positions(joints_rad)
                ee = ik.get_ee_position()
                print(f"EE position: x={ee[0]:.3f}, y={ee[1]:.3f}, z={ee[2]:.3f}")
            elif cmd == 'l':
                ik_reset(robot, ik)
            elif cmd == '+3':
                wrist_flex += 5
                joints = robot.get_joint_positions_radians()
                joints[3] = np.deg2rad(wrist_flex)
                robot.send_action(joints, 1.0)
            elif cmd == '-3':
                wrist_flex -= 5
                joints = robot.get_joint_positions_radians()
                joints[3] = np.deg2rad(wrist_flex)
                robot.send_action(joints, 1.0)
            elif cmd == '+4':
                wrist_roll += 5
                joints = robot.get_joint_positions_radians()
                joints[4] = np.deg2rad(wrist_roll)
                robot.send_action(joints, 1.0)
            elif cmd == '-4':
                wrist_roll -= 5
                joints = robot.get_joint_positions_radians()
                joints[4] = np.deg2rad(wrist_roll)
                robot.send_action(joints, 1.0)
            elif cmd.startswith('3 '):
                try:
                    wrist_flex = float(cmd.split()[1])
                    joints = robot.get_joint_positions_radians()
                    joints[3] = np.deg2rad(wrist_flex)
                    robot.send_action(joints, 1.0)
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd.startswith('4 '):
                try:
                    wrist_roll = float(cmd.split()[1])
                    joints = robot.get_joint_positions_radians()
                    joints[4] = np.deg2rad(wrist_roll)
                    robot.send_action(joints, 1.0)
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd:
                print("Unknown command. Use: 3 <angle>, 4 <angle>, +3, -3, +4, -4, r, l, q")

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
        robot.disconnect()


if __name__ == "__main__":
    main()
