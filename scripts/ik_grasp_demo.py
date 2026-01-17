#!/usr/bin/env python3
"""IK-based grasp demo with wrist camera recording.

Performs a top-down pick sequence using IK control and records wrist cam footage.

Based on pick-101's test_topdown_pick.py 4-step sequence:
1. Move above block (30mm height), gripper open
2. Move down to block, gripper open
3. Close gripper
4. Lift back up

Usage:
    # Dry run (mock robot and camera)
    uv run python scripts/ik_grasp_demo.py --dry_run

    # Real robot with default cube position
    uv run python scripts/ik_grasp_demo.py

    # With custom cube position and output video
    uv run python scripts/ik_grasp_demo.py --cube_x 0.25 --cube_y 0.0 --output grasp_demo.mp4
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

from src.deploy.camera import CameraPreprocessor
from src.deploy.robot import SO101Robot, MockSO101Robot
from src.deploy.controllers import IKController


# Constants from pick-101's test_topdown_pick.py (adjusted for real robot)
HEIGHT_OFFSET = 0.06      # 60mm above grasp position (for "above" position)
GRASP_Z_OFFSET = 0.005    # 5mm above cube center for grasp (sim reference)
GRASP_HEIGHT_SAFETY = 0.05  # Extra 50mm clearance for real robot (grasp at 70mm instead of 20mm)
CUBE_Z = 0.015            # Cube center height (30mm cube on table)
FINGER_WIDTH_OFFSET = -0.015  # Y offset to center grip on cube

GRIPPER_OPEN = 0.3        # Partially open (like pick-101)
GRIPPER_CLOSED = -1.0     # Closed

# Joint offset correction (from kinematic verification)
ELBOW_FLEX_OFFSET_RAD = np.deg2rad(-12.5)


def apply_joint_offset(joints: np.ndarray) -> np.ndarray:
    """Apply calibration offset correction to joint readings."""
    corrected = joints.copy()
    corrected[2] += ELBOW_FLEX_OFFSET_RAD
    return corrected


def move_to_position(robot, ik, target_pos, gripper_action, num_steps=100, dt=0.05, record_callback=None):
    """Move robot to target EE position using IK with wrist locked."""
    for step in range(num_steps):
        current_joints_raw = robot.get_joint_positions_radians()
        current_joints = apply_joint_offset(current_joints_raw)

        # Lock wrist joints at π/2 for top-down (matches MuJoCo convention)
        current_joints[3] = np.pi / 2
        current_joints[4] = np.pi / 2

        # Multiple IK iterations for convergence
        for _ in range(3):
            target_joints = ik.compute_ik(target_pos, current_joints, locked_joints=[3, 4])
            current_joints = target_joints

        # Ensure wrist stays locked
        target_joints[3] = np.pi / 2
        target_joints[4] = np.pi / 2

        # Undo offset for robot command
        target_joints[2] -= ELBOW_FLEX_OFFSET_RAD

        robot.send_action(target_joints, gripper_action)

        # Record frame during motion
        if record_callback:
            record_callback()

        time.sleep(dt)

        # Check convergence
        ik.sync_joint_positions(apply_joint_offset(robot.get_joint_positions_radians()))
        ee_pos = ik.get_ee_position()
        error = np.linalg.norm(target_pos - ee_pos)
        if error < 0.005:  # Within 5mm
            break

    return ee_pos


def center_crop_square(image: np.ndarray) -> np.ndarray:
    """Center crop to square."""
    h, w = image.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return image[y_start:y_start + size, x_start:x_start + size]


def main():
    parser = argparse.ArgumentParser(description="IK grasp demo with wrist cam recording")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--camera_index", type=int, default=0, help="Wrist camera index")
    parser.add_argument("--cube_x", type=float, default=0.25, help="Cube X position (meters)")
    parser.add_argument("--cube_y", type=float, default=0.0, help="Cube Y position (meters)")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: auto-generated)")
    parser.add_argument("--dry_run", action="store_true", help="Run without real robot/camera")
    parser.add_argument("--fps", type=float, default=20.0, help="Recording FPS")
    parser.add_argument("--gripper_open", type=float, default=1.0, help="Gripper open position (-1.0=closed, 1.0=fully open, default=1.0)")
    parser.add_argument("--half_open", action="store_true", help="Use half-open gripper (0.0 = 50%% physical)")
    args = parser.parse_args()

    # Set gripper open position from argument (--half_open overrides --gripper_open)
    gripper_open = 0.0 if args.half_open else args.gripper_open

    print("=" * 60)
    print("IK Grasp Demo with Wrist Cam Recording")
    print("=" * 60)
    print(f"Gripper open position: {gripper_open}")

    # Initialize robot
    print("\n[1/3] Initializing robot...")
    if args.dry_run:
        robot = MockSO101Robot(port=args.robot_port)
        robot.connect()
        print("  [DRY RUN] Using mock robot")
    else:
        robot = SO101Robot(port=args.robot_port)
        if not robot.connect():
            print("Failed to connect to robot. Exiting.")
            return

    # Initialize IK controller
    print("\n[2/3] Initializing IK controller...")
    try:
        ik = IKController()
        print(f"  IK controller ready")
    except Exception as e:
        print(f"Failed to initialize IK: {e}")
        robot.disconnect()
        return

    # Initialize camera and video writer
    print("\n[3/3] Initializing camera...")
    camera = None
    video_writer = None
    frame_size = (480, 480)

    if not args.dry_run:
        cap = cv2.VideoCapture(args.camera_index)
        if cap.isOpened():
            # Warm up camera
            for _ in range(10):
                cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Camera opened: {w}x{h}")
            camera = cap
        else:
            print("  WARNING: Camera not available, continuing without recording")

    # Setup video output
    if camera is not None:
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ik_grasp_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, frame_size)
        print(f"  Recording to: {output_path}")

    # Safe positions
    RESET_JOINTS = np.zeros(5)  # All motors in middle of range (arm extended forward)
    REST_JOINTS = np.array([-0.1756, -1.8619, 1.7922, 0.7840, -0.2187])  # Folded rest

    def record_frame():
        """Capture and record a frame."""
        if camera is not None and video_writer is not None:
            ret, frame = camera.read()
            if ret:
                cropped = center_crop_square(frame)
                resized = cv2.resize(cropped, frame_size)
                video_writer.write(resized)

    def record_steps(num_steps, delay=0.05):
        """Record multiple frames during a pause."""
        for _ in range(num_steps):
            record_frame()
            time.sleep(delay)

    print("\n" + "=" * 60)
    print("Starting grasp sequence...")
    print("=" * 60)

    cube_pos = np.array([args.cube_x, args.cube_y, CUBE_Z])
    print(f"Cube position: {cube_pos}")

    try:
        # Step 0: Go to reset position (all motors in middle of range)
        print("\n[Step 0] Moving to reset position...")
        print(f"  Target (rad): {RESET_JOINTS}")
        print(f"  Target (deg): {np.rad2deg(RESET_JOINTS)}")
        current_before = robot.get_joint_positions_radians()
        print(f"  Current (deg): {np.rad2deg(current_before)}")
        robot.send_action(RESET_JOINTS, gripper_open)
        time.sleep(3.0)  # Wait for motors to reach position
        current_after = robot.get_joint_positions_radians()
        print(f"  After (deg): {np.rad2deg(current_after)}")
        record_steps(30)  # Record 1.5s at safe position

        # Set wrist joints to top-down (matches MuJoCo convention)
        print("  Setting top-down wrist orientation...")
        topdown_joints = robot.get_joint_positions_radians().copy()
        topdown_joints[3] = np.pi / 2
        topdown_joints[4] = np.pi / 2
        robot.send_action(topdown_joints, gripper_open)
        record_steps(20)

        # Step 1: Move above block with gripper open
        above_pos = cube_pos.copy()
        above_pos[2] += GRASP_Z_OFFSET + HEIGHT_OFFSET  # 50mm above ground
        above_pos[1] += FINGER_WIDTH_OFFSET
        print(f"\n[Step 1] Moving above block...")
        print(f"  Target: {above_pos}")

        ee_pos = move_to_position(robot, ik, above_pos, gripper_open, num_steps=100, record_callback=record_frame)
        print(f"  Reached: {ee_pos}")
        record_steps(20)

        # Step 2: Move down to block with gripper open
        grasp_pos = cube_pos.copy()
        grasp_pos[2] += GRASP_Z_OFFSET + GRASP_HEIGHT_SAFETY  # 35mm above ground (safe grasp height for real robot)
        grasp_pos[1] += FINGER_WIDTH_OFFSET
        print(f"\n[Step 2] Moving down to block...")
        print(f"  Target: {grasp_pos}")

        ee_pos = move_to_position(robot, ik, grasp_pos, gripper_open, num_steps=80, record_callback=record_frame)
        print(f"  Reached: {ee_pos}")
        record_steps(20)

        # Save grasp position joints
        grasp_joints = robot.get_joint_positions_radians()

        # Step 3: Close gripper - keep arm at grasp position
        print(f"\n[Step 3] Closing gripper...")
        for grip in np.linspace(gripper_open, GRIPPER_CLOSED, 20):
            robot.send_action(grasp_joints, grip)
            record_frame()
            time.sleep(0.05)
        record_steps(20)
        print(f"  Gripper closed to {GRIPPER_CLOSED}")

        # Step 4: Lift up
        lift_pos = cube_pos.copy()
        lift_pos[2] += HEIGHT_OFFSET + 0.05  # Lift to 80mm
        lift_pos[1] += FINGER_WIDTH_OFFSET
        print(f"\n[Step 4] Lifting...")
        print(f"  Target: {lift_pos}")

        ee_pos = move_to_position(robot, ik, lift_pos, GRIPPER_CLOSED, num_steps=80, record_callback=record_frame)
        print(f"  Reached: {ee_pos}")
        record_steps(40)  # Hold and record

        # Save lift position joints for release step
        lift_joints = robot.get_joint_positions_radians()

        # Step 5: Open gripper (release) - keep arm at lift position
        print(f"\n[Step 5] Releasing...")
        for grip in np.linspace(GRIPPER_CLOSED, gripper_open, 20):
            robot.send_action(lift_joints, grip)
            record_frame()
            time.sleep(0.05)
        record_steps(20)

        print("\n" + "=" * 60)
        print("Grasp sequence complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Safe return sequence
        print("\nSafe return sequence...")

        # Step 1: Go to reset position first (all motors in middle of range)
        print("  Moving to reset position...")
        robot.send_action(RESET_JOINTS, gripper_open)
        time.sleep(2.0)
        record_steps(20)

        # Step 2: Go to rest position
        print("  Moving to rest position...")
        robot.send_action(REST_JOINTS, -1.0)
        time.sleep(2.0)
        record_steps(20)

        # Cleanup
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved: {output_path}")
        if camera is not None:
            camera.release()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
