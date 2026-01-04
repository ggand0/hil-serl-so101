#!/usr/bin/env python3
"""Move robot to training initial position for marking cube location.

Use this to mark where the cube should be placed on your table.
The robot will move to the position where it expects the cube during inference.

Usage:
    uv run python scripts/ik_reset_position.py                    # Default position
    uv run python scripts/ik_reset_position.py --cube_x 0.28      # Adjust X
    uv run python scripts/ik_reset_position.py --dry_run          # Mock robot
"""

import argparse
import time
from pathlib import Path

import numpy as np

# Add project root to path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deploy.robot import SO101Robot, MockSO101Robot
from src.deploy.controllers import IKController


# Training initial position constants (from curriculum_stage=3 in lift_cube.py)
FINGER_WIDTH_OFFSET = -0.015  # Static finger offset from gripper center
GRASP_Z_OFFSET = 0.005
HEIGHT_OFFSET = 0.03  # Start 3cm above grasp height
CUBE_Z = 0.015  # Cube resting height on table


def main():
    parser = argparse.ArgumentParser(description="Move robot to training initial position")
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Robot serial port",
    )
    parser.add_argument(
        "--cube_x",
        type=float,
        default=0.25,
        help="Expected cube X position (meters, default 0.25)",
    )
    parser.add_argument(
        "--cube_y",
        type=float,
        default=0.0,
        help="Expected cube Y position (meters, default 0.0)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use mock robot (no real hardware)",
    )
    parser.add_argument(
        "--lower",
        action="store_true",
        help="Lower gripper to grasp height (touch cube level)",
    )
    parser.add_argument(
        "--record_only",
        action="store_true",
        help="Just read and print current robot position (don't move)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("IK Reset Position Tool")
    print("=" * 60)

    # Record-only mode: just read current position
    if args.record_only:
        print("\n[RECORD MODE] Reading current robot position...")
        if args.dry_run:
            robot = MockSO101Robot(port=args.robot_port)
            robot.connect()
        else:
            robot = SO101Robot(port=args.robot_port)
            if not robot.connect():
                print("Failed to connect to robot.")
                return

        try:
            ik = IKController()
        except Exception as e:
            print(f"Failed to initialize IK: {e}")
            robot.disconnect()
            return

        joints = robot.get_joint_positions_radians()
        ik.sync_joint_positions(joints)
        ee_pos = ik.get_ee_position()
        gripper = robot.get_gripper_position()

        print("\n" + "=" * 60)
        print("CURRENT ROBOT POSITION")
        print("=" * 60)
        print(f"\nJoints (rad): {joints}")
        print(f"EE position:  {ee_pos}")
        print(f"Gripper:      {gripper:.3f}")
        print(f"\nCopy-paste:")
        joints_str = ", ".join(f"{float(j):.4f}" for j in joints)
        ee_str = ", ".join(f"{float(e):.4f}" for e in ee_pos)
        print(f"  HOME_JOINTS = np.array([{joints_str}])")
        print(f"  HOME_EE = np.array([{ee_str}])")

        robot.disconnect()
        print("\nDone.")
        return

    print(f"Cube position: X={args.cube_x:.3f}m, Y={args.cube_y:.3f}m")

    # Calculate target positions
    above_target = np.array([
        args.cube_x,
        args.cube_y + FINGER_WIDTH_OFFSET,
        CUBE_Z + GRASP_Z_OFFSET + HEIGHT_OFFSET  # ~0.05m
    ])

    grasp_target = np.array([
        args.cube_x,
        args.cube_y + FINGER_WIDTH_OFFSET,
        CUBE_Z + GRASP_Z_OFFSET  # ~0.02m (at cube level)
    ])

    print(f"Above-cube position: {above_target}")
    print(f"Grasp position: {grasp_target}")

    # Initialize robot
    print("\n[1/3] Connecting to robot...")
    if args.dry_run:
        robot = MockSO101Robot(port=args.robot_port)
        robot.connect()
        print("  [DRY RUN] Using mock robot")
    else:
        robot = SO101Robot(port=args.robot_port)
        if not robot.connect():
            print("Failed to connect to robot. Exiting.")
            return
        print(f"  Connected on {args.robot_port}")

    # Initialize IK controller
    print("\n[2/3] Initializing IK controller...")
    try:
        ik = IKController()
        print(f"  IK controller ready")
    except Exception as e:
        print(f"  Failed to initialize IK: {e}")
        robot.disconnect()
        return

    # Safe starting pose (extended forward) and rest pose (folded near base)
    SAFE_JOINTS = np.zeros(5)  # Extended forward - safe for IK movements
    REST_JOINTS = np.array([-0.2424, -1.8040, 1.6582, 0.7309, -0.0629])  # Folded rest

    def safe_return():
        """Safe return sequence: lift up first, then go to rest position."""
        print("\nSafe return sequence...")

        # Step 1: Lift up to safe height (keep wrist orientation)
        print("  Lifting to safe height...")
        current_joints = robot.get_joint_positions_radians()
        ik.sync_joint_positions(current_joints)
        current_ee = ik.get_ee_position()
        safe_height_target = current_ee.copy()
        safe_height_target[2] = 0.15  # Lift to 15cm

        for step in range(40):
            current_joints = robot.get_joint_positions_radians()
            current_joints[3] = np.pi / 2
            current_joints[4] = np.pi / 2
            target_joints = ik.compute_ik(safe_height_target, current_joints, locked_joints=[3, 4])
            target_joints[3] = np.pi / 2
            target_joints[4] = np.pi / 2
            robot.send_action(target_joints, 1.0)
            time.sleep(0.05)

            ik.sync_joint_positions(robot.get_joint_positions_radians())
            ee_pos = ik.get_ee_position()
            if ee_pos[2] > 0.12:  # High enough
                break

        print(f"  Lifted to: {ik.get_ee_position()}")
        time.sleep(0.3)

        # Step 2: Return to rest position with gripper closed
        print("  Returning to rest position...")
        robot.send_action(REST_JOINTS, -1.0)  # Close gripper at rest
        time.sleep(1.0)

    try:
        print("\n[3/3] Moving to positions...")
        print("  Step 1: Safe extended position...")
        robot.send_action(SAFE_JOINTS, 1.0)  # Open gripper
        time.sleep(1.5)

        # Get current state
        joint_pos = robot.get_joint_positions_radians()
        ik.sync_joint_positions(joint_pos)
        ee_pos = ik.get_ee_position()
        print(f"  Current EE: {ee_pos}")

        # Step 1.5: Set wrist joints to π/2 for top-down orientation (matches training)
        print("  Step 1.5: Setting top-down wrist orientation (joints 3,4 = π/2)...")
        topdown_joints = robot.get_joint_positions_radians().copy()
        topdown_joints[3] = np.pi / 2  # wrist_pitch
        topdown_joints[4] = np.pi / 2  # wrist_roll
        robot.send_action(topdown_joints, 1.0)
        time.sleep(1.0)

        joint_pos = robot.get_joint_positions_radians()
        ik.sync_joint_positions(joint_pos)
        ee_pos = ik.get_ee_position()
        print(f"  After wrist setup EE: {ee_pos}")
        print(f"  Joints: {joint_pos}")

        # Move to above-cube position (lock wrist joints at π/2)
        print("  Step 2: Moving above cube position (wrist locked)...")
        target = above_target
        for step in range(100):
            current_joints = robot.get_joint_positions_radians()
            # Lock wrist joints 3,4 at π/2 during IK
            current_joints[3] = np.pi / 2
            current_joints[4] = np.pi / 2
            # Multiple IK iterations per step for better convergence
            for _ in range(3):
                target_joints = ik.compute_ik(target, current_joints, locked_joints=[3, 4])
                current_joints = target_joints
            # Ensure wrist stays locked
            target_joints[3] = np.pi / 2
            target_joints[4] = np.pi / 2
            robot.send_action(target_joints, 1.0)
            time.sleep(0.05)

            ik.sync_joint_positions(robot.get_joint_positions_radians())
            ee_pos = ik.get_ee_position()
            error = np.linalg.norm(target - ee_pos)
            if step % 20 == 0:
                print(f"    Step {step}: EE={ee_pos}, error={error*1000:.1f}mm")
            if error < 0.008:  # Within 8mm
                break

        print(f"  Reached: {ee_pos}")
        print(f"  Error: {np.linalg.norm(target - ee_pos)*1000:.1f}mm")
        print(f"  Joints: {robot.get_joint_positions_radians()}")

        if args.lower:
            print("\n  Step 3: Lowering to grasp height (wrist locked)...")
            time.sleep(0.5)
            target = grasp_target
            for step in range(60):
                current_joints = robot.get_joint_positions_radians()
                current_joints[3] = np.pi / 2
                current_joints[4] = np.pi / 2
                for _ in range(3):
                    target_joints = ik.compute_ik(target, current_joints, locked_joints=[3, 4])
                    current_joints = target_joints
                target_joints[3] = np.pi / 2
                target_joints[4] = np.pi / 2
                robot.send_action(target_joints, 1.0)
                time.sleep(0.05)

                ik.sync_joint_positions(robot.get_joint_positions_radians())
                ee_pos = ik.get_ee_position()
                error = np.linalg.norm(target - ee_pos)
                if step % 20 == 0:
                    print(f"    Step {step}: EE={ee_pos}, error={error*1000:.1f}mm")
                if error < 0.008:
                    break

            print(f"  Reached: {ee_pos}")
            print(f"  Error: {np.linalg.norm(target - ee_pos)*1000:.1f}mm")
            print(f"  Joints: {robot.get_joint_positions_radians()}")

        # Record final position
        final_joints = robot.get_joint_positions_radians()
        ik.sync_joint_positions(final_joints)
        final_ee = ik.get_ee_position()

        # Hold position
        print("\n" + "=" * 60)
        print("HOLDING POSITION - Mark the cube location on your table!")
        print("The gripper center is at the expected cube position.")
        print("=" * 60)
        print(f"\nFinal EE position: {final_ee}")
        print(f"Final joints (rad): {final_joints}")
        print(f"\nTo use this position:")
        print(f"  joints = np.array({list(final_joints)})")
        print(f"  ee_pos = np.array({list(final_ee)})")
        print("\nPress Enter to return to home, or Ctrl+C to exit...")

        try:
            input()
        except KeyboardInterrupt:
            print("\nInterrupted")

    except KeyboardInterrupt:
        print("\nInterrupted during movement")

    finally:
        safe_return()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
