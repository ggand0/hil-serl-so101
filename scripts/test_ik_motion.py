#!/usr/bin/env python3
"""Test IK-controlled motion on real SO-101.

Moves the end-effector in small increments using IK.
No policy involved - just tests the IK → robot pipeline.

Usage:
    uv run python scripts/test_ik_motion.py --dry_run   # Mock robot
    uv run python scripts/test_ik_motion.py             # Real robot
"""

import argparse
import time
from pathlib import Path
import sys

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deploy.robot import SO101Robot, MockSO101Robot
from src.deploy.controllers import IKController


def move_delta(robot, ik, delta_xyz, gripper_open, dry_run, delay=0.05):
    """Move EE by a delta using IK (doesn't depend on absolute calibration)."""
    current_joints = robot.get_joint_positions_radians()

    # Use IK Jacobian to compute joint delta for Cartesian delta
    # This works because Jacobian is local and doesn't need global calibration
    target_joints = ik.cartesian_to_joints(
        delta_xyz / 0.02,  # Normalize assuming 2cm scale
        current_joints,
        action_scale=0.02,
    )

    gripper_val = 1.0 if gripper_open else -1.0
    if not dry_run:
        robot.send_action(target_joints, gripper_val)
        time.sleep(delay)

    return robot.get_joint_positions_radians()


def move_to_position(robot, ik, target_pos, gripper_open, dry_run, steps=20, delay=0.05):
    """Move EE to target position using iterative IK.

    NOTE: This requires calibrated FK. If FK doesn't match reality,
    use move_delta() instead for relative movements.
    """
    for _ in range(steps):
        current_joints = robot.get_joint_positions_radians()
        ik.sync_joint_positions(current_joints)
        current_pos = ik.get_ee_position()

        # Compute delta
        delta = target_pos - current_pos
        dist = np.linalg.norm(delta)

        if dist < 0.005:  # 5mm threshold
            break

        # Clamp step size
        max_step = 0.02
        if dist > max_step:
            delta = delta / dist * max_step

        # IK
        target_joints = ik.compute_ik(current_pos + delta, current_joints)

        # Send
        gripper_val = 1.0 if gripper_open else -1.0
        if not dry_run:
            robot.send_action(target_joints, gripper_val)
            time.sleep(delay)

    # Final position
    ik.sync_joint_positions(robot.get_joint_positions_radians())
    return ik.get_ee_position()


def pick_here(robot, ik, dry_run, down_distance=0.05, lift_distance=0.08):
    """Execute pick sequence at CURRENT position (delta-based, no calibration needed).

    Lowers gripper, closes, and lifts.

    Args:
        down_distance: How far to lower (meters)
        lift_distance: How far to lift after grabbing (meters)
    """
    print(f"\n>>> Pick sequence (down={down_distance}m, lift={lift_distance}m)")

    # 1. Open gripper
    print("  1. Opening gripper...")
    current_joints = robot.get_joint_positions_radians()
    if not dry_run:
        robot.send_action(current_joints, 1.0)  # 1.0 = open
        time.sleep(0.3)

    # 2. Lower (negative Z delta)
    print(f"  2. Lowering {down_distance*100:.0f}cm...")
    steps = int(down_distance / 0.01)  # 1cm per step
    for _ in range(steps):
        move_delta(robot, ik, np.array([0, 0, -0.01]), gripper_open=True, dry_run=dry_run)
    print("     Done")

    # 3. Close gripper
    print("  3. Closing gripper...")
    current_joints = robot.get_joint_positions_radians()
    if not dry_run:
        robot.send_action(current_joints, -1.0)  # -1.0 = closed
        time.sleep(0.5)

    # 4. Lift (positive Z delta)
    print(f"  4. Lifting {lift_distance*100:.0f}cm...")
    steps = int(lift_distance / 0.01)
    for _ in range(steps):
        move_delta(robot, ik, np.array([0, 0, 0.01]), gripper_open=False, dry_run=dry_run)
    print("     Done")

    print(">>> Pick complete!")
    return True


def pick_at_position(robot, ik, target_xyz, dry_run):
    """Execute pick sequence at given XYZ position.

    WARNING: Requires calibrated FK. Use pick_here() for uncalibrated robots.
    """
    print(f"\n>>> Pick sequence at [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")

    # Heights
    hover_height = target_xyz[2] + 0.05  # 5cm above target

    # 1. Open gripper
    print("  1. Opening gripper...")
    current_joints = robot.get_joint_positions_radians()
    if not dry_run:
        robot.send_action(current_joints, 1.0)  # 1.0 = open
        time.sleep(0.3)

    # 2. Move above target
    hover_pos = np.array([target_xyz[0], target_xyz[1], hover_height])
    print(f"  2. Moving to hover position {hover_pos}...")
    ee_pos = move_to_position(robot, ik, hover_pos, gripper_open=True, dry_run=dry_run)
    print(f"     Reached: {ee_pos}")

    # 3. Lower to target
    print(f"  3. Lowering to target...")
    ee_pos = move_to_position(robot, ik, target_xyz, gripper_open=True, dry_run=dry_run)
    print(f"     Reached: {ee_pos}")

    # 4. Close gripper
    print("  4. Closing gripper...")
    current_joints = robot.get_joint_positions_radians()
    if not dry_run:
        robot.send_action(current_joints, -1.0)  # -1.0 = closed
        time.sleep(0.5)

    # 5. Lift
    lift_pos = np.array([target_xyz[0], target_xyz[1], hover_height])
    print(f"  5. Lifting...")
    ee_pos = move_to_position(robot, ik, lift_pos, gripper_open=False, dry_run=dry_run)
    print(f"     Reached: {ee_pos}")

    print(">>> Pick complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test IK motion on SO-101")
    parser.add_argument("--dry_run", action="store_true", help="Use mock robot")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    args = parser.parse_args()

    # Initialize robot
    if args.dry_run:
        robot = MockSO101Robot(port=args.port)
    else:
        robot = SO101Robot(port=args.port)

    if not robot.connect():
        print("Failed to connect to robot")
        return

    # Initialize IK
    try:
        ik = IKController()
    except Exception as e:
        print(f"Failed to init IK: {e}")
        robot.disconnect()
        return

    print("\n=== IK Motion Test ===")
    print("Commands: w/s = forward/back, a/d = left/right, q/e = up/down")
    print("          g = toggle gripper, r = reset, x = exit")
    print("          h = pick HERE (lower, grab, lift - no calibration needed)")
    print("          p = pick at XYZ (needs calibrated FK)\n")

    # Get initial state
    joint_pos = robot.get_joint_positions_radians()
    ik.sync_joint_positions(joint_pos)
    ee_pos = ik.get_ee_position()
    gripper_open = True

    print(f"Initial EE: {ee_pos}")
    print(f"Initial joints: {joint_pos}")

    step_size = 0.02  # 2cm per keypress

    try:
        while True:
            cmd = input("\nCommand: ").strip().lower()

            if cmd == 'x':
                break
            elif cmd == 'r':
                # Reset to home
                joint_pos = np.zeros(5)
                if not args.dry_run:
                    robot.send_action(joint_pos, 1.0 if gripper_open else -1.0)
                    time.sleep(0.5)
                ik.sync_joint_positions(joint_pos)
                ee_pos = ik.get_ee_position()
                print(f"Reset to home. EE: {ee_pos}")
                continue
            elif cmd == 'g':
                gripper_open = not gripper_open
                # Fixed: -1.0 = closed, 1.0 = open
                gripper_val = 1.0 if gripper_open else -1.0
                current_joints = robot.get_joint_positions_radians()
                if not args.dry_run:
                    robot.send_action(current_joints, gripper_val)
                print(f"Gripper: {'open' if gripper_open else 'closed'}")
                continue
            elif cmd == 'h':
                # Pick here - delta based, no calibration needed
                pick_here(robot, ik, args.dry_run)
                gripper_open = False
                continue
            elif cmd == 'p':
                # Pick at position (needs calibrated FK)
                try:
                    pos_str = input("Enter target X Y Z (e.g., 0.2 0.0 0.05): ").strip()
                    coords = [float(x) for x in pos_str.split()]
                    if len(coords) != 3:
                        print("Need exactly 3 coordinates")
                        continue
                    target = np.array(coords)
                    pick_at_position(robot, ik, target, args.dry_run)
                    gripper_open = False  # Gripper is now closed
                except ValueError:
                    print("Invalid coordinates")
                continue

            # Parse movement
            delta = np.zeros(3)
            if cmd == 'w':
                delta[0] = step_size  # +X forward
            elif cmd == 's':
                delta[0] = -step_size
            elif cmd == 'a':
                delta[1] = step_size  # +Y left
            elif cmd == 'd':
                delta[1] = -step_size
            elif cmd == 'q':
                delta[2] = step_size  # +Z up
            elif cmd == 'e':
                delta[2] = -step_size
            else:
                print("Unknown command")
                continue

            # Compute IK
            current_joints = robot.get_joint_positions_radians()
            target_joints = ik.cartesian_to_joints(
                delta / step_size,  # Normalize to [-1, 1]
                current_joints,
                action_scale=step_size,
            )

            # Send to robot
            gripper_val = 1.0 if gripper_open else -1.0
            if not args.dry_run:
                robot.send_action(target_joints, gripper_val)
                time.sleep(0.1)

            # Update state
            joint_pos = robot.get_joint_positions_radians()
            ik.sync_joint_positions(joint_pos)
            ee_pos = ik.get_ee_position()

            print(f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]  "
                  f"Joints: {joint_pos[:3].round(2)}...")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
