#!/usr/bin/env python3
"""Test different IK reset heights using the actual IK reset motion from training."""

import signal
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from lerobot.robots.so101_follower import SO101FollowerEndEffectorConfig, SO101FollowerEndEffector
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Heights to test (z coordinate in meters)
heights = [0.03, 0.05, 0.07, 0.10]

# Joint names in order (matching gym_manipulator.py _IK_MOTOR_NAMES)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Height offset for approach (matching gym_manipulator.py)
HEIGHT_OFFSET = 0.07  # 7cm above target for approach

# REST_JOINTS from actor.py/rl_inference.py - safe resting position
# REST_JOINTS_RAD = np.array([-0.2424, -1.8040, 1.6582, 0.7309, -0.0629])
REST_JOINTS_DEG = np.array([-13.89, -103.36, 95.01, 41.88, -3.60])

# SAFE_JOINTS - intermediate safe position (all zeros, arm extended forward)
SAFE_JOINTS_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Global robot reference for signal handler
_robot = None
_camera = None
_cleanup_done = False

def cleanup():
    """
    Safe return sequence matching actor.py safe_return_to_home():
    1. First lift to safe height (15cm) using IK
    2. Then move to REST_JOINTS position
    Does NOT disconnect to keep torque enabled.
    """
    global _robot, _camera, _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    if _robot is not None and _robot.is_connected:
        print("\n=== Safe Return Sequence ===")

        # Step 1: Lift to safe height (15cm) using IK
        print("Step 1: Lifting to safe height (15cm)...")
        try:
            obs = _robot.get_observation()
            current_deg = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])
            current_rad = np.deg2rad(current_deg)

            _robot._sync_mujoco(current_rad)
            current_ee = _robot._get_ee_position()

            # Target: lift to 15cm height
            safe_height_target = current_ee.copy()
            safe_height_target[2] = 0.15

            for step in range(40):
                obs = _robot.get_observation()
                current_deg = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])
                current_rad = np.deg2rad(current_deg)

                # Lock wrist orientation
                current_rad[3] = np.pi / 2   # wrist_flex
                current_rad[4] = -np.pi / 2  # wrist_roll

                # Compute IK with locked wrist
                target_rad = _robot._compute_ik(safe_height_target, current_rad)
                target_rad[3] = np.pi / 2
                target_rad[4] = -np.pi / 2

                target_deg = np.rad2deg(target_rad)

                # Clamp delta to max 10° per step
                delta_deg = np.clip(target_deg - current_deg, -10, 10)
                target_deg = current_deg + delta_deg
                target_deg = clamp_degrees(target_deg)

                action_dict = {f"{name}.pos": target_deg[i] for i, name in enumerate(JOINT_NAMES)}
                action_dict["gripper.pos"] = 50.0
                _robot.send_action(action_dict)
                busy_wait(0.05)

                # Check if high enough
                _robot._sync_mujoco(np.deg2rad(target_deg))
                ee_pos = _robot._get_ee_position()
                if ee_pos[2] > 0.12:
                    print(f"  Lifted to height: {ee_pos[2]*100:.1f}cm")
                    break

            busy_wait(0.3)
        except Exception as e:
            print(f"  Lift failed ({e}), going directly to rest...")

        # Step 2: Interpolate to REST_JOINTS position (gradual movement like ik_grasp_demo)
        print("Step 2: Interpolating to rest position...")
        try:
            obs = _robot.get_observation()
            current_deg = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])
            rest_deg = clamp_degrees(REST_JOINTS_DEG.copy())

            # Gradual interpolation over 20 steps (2 seconds)
            for i in range(20):
                alpha = (i + 1) / 20
                interp_deg = (1 - alpha) * current_deg + alpha * rest_deg
                action_dict = {f"{name}.pos": interp_deg[j] for j, name in enumerate(JOINT_NAMES)}
                action_dict["gripper.pos"] = -50.0  # Close gripper
                _robot.send_action(action_dict)
                time.sleep(0.1)

            # Final position
            action_dict = {f"{name}.pos": rest_deg[i] for i, name in enumerate(JOINT_NAMES)}
            action_dict["gripper.pos"] = -50.0
            _robot.send_action(action_dict)
            time.sleep(1.0)

            print("Robot at rest position. Torque remains ON.")
        except Exception as e:
            print(f"Error moving to rest: {e}")

        # DO NOT disconnect - keep torque enabled for safety

    if _camera is not None:
        cv2.destroyAllWindows()
        try:
            _camera.disconnect()
            print("Camera disconnected.")
        except Exception:
            pass

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nCtrl+C detected!")
    cleanup()
    print("\nPress Enter to exit (torque will be disabled)...")
    try:
        input()
    except EOFError:
        pass
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def busy_wait(seconds: float):
    """Busy wait for precise timing."""
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

def clamp_degrees(joints_deg: np.ndarray) -> np.ndarray:
    """Clamp joint angles to valid range."""
    limits = [(-180, 180), (-90, 90), (-90, 90), (-90, 90), (-180, 180)]
    clamped = joints_deg.copy()
    for i, (lo, hi) in enumerate(limits):
        clamped[i] = np.clip(clamped[i], lo, hi)
    return clamped

def update_camera_preview(camera):
    """Update the camera preview window."""
    try:
        frame = camera.async_read()
        if frame is not None:
            # Apply same crop as training: [0, 80, 480, 480]
            h, w = frame.shape[:2]
            crop_x, crop_y, crop_w, crop_h = 0, 80, 480, 480
            if crop_y + crop_h <= h and crop_x + crop_w <= w:
                cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            else:
                cropped = frame

            # Convert RGB to BGR for OpenCV display
            display = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

            # Resize to 256x256 for display
            display = cv2.resize(display, (256, 256))

            # Add height indicator text
            cv2.putText(display, "Gripper Cam (cropped)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Camera Preview", display)
            cv2.waitKey(1)
    except Exception as e:
        pass  # Ignore camera errors during preview

def ik_reset_motion(robot, camera, target_z: float):
    """
    Perform the full IK reset motion matching gym_manipulator.py ResetWrapper.
    """
    target_ee = np.array([0.25, 0.0, target_z])

    print(f"\n=== IK Reset to z={target_z*100:.0f}cm ===")
    print(f"Target EE: {target_ee}")

    # Step 1: Move to SAFE_JOINTS (all zeros)
    print("Step 1: Moving to SAFE_JOINTS (all zeros)")
    safe_joints_deg = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    action_dict = {f"{name}.pos": safe_joints_deg[i] for i, name in enumerate(JOINT_NAMES)}
    action_dict["gripper.pos"] = 50.0
    robot.send_action(action_dict)
    for _ in range(15):  # 1.5s with camera updates
        update_camera_preview(camera)
        busy_wait(0.1)

    # Step 2: Set wrist to top-down orientation
    print("Step 2: Setting wrist to top-down (90°)")
    obs = robot.get_observation()
    current_deg = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])
    current_deg[3] = 90.0  # wrist_flex
    current_deg[4] = 90.0  # wrist_roll
    current_deg = clamp_degrees(current_deg)
    action_dict = {f"{name}.pos": current_deg[i] for i, name in enumerate(JOINT_NAMES)}
    action_dict["gripper.pos"] = 50.0
    robot.send_action(action_dict)
    for _ in range(10):  # 1.0s with camera updates
        update_camera_preview(camera)
        busy_wait(0.1)

    # Step 3: Move ABOVE target position
    above_target = target_ee.copy()
    above_target[2] += HEIGHT_OFFSET
    print(f"Step 3: Moving above target ({above_target})")
    ik_move(robot, camera, above_target)

    # Step 4: Lower to target position
    print(f"Step 4: Lowering to target ({target_ee})")
    ik_move(robot, camera, target_ee)

    # Verify final position
    obs = robot.get_observation()
    final_joints = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])
    robot._sync_mujoco(np.deg2rad(final_joints))
    final_ee = robot._get_ee_position()
    error = np.linalg.norm(target_ee - final_ee)

    print(f"\nFinal EE position: {final_ee}")
    print(f"Height above table: {final_ee[2]*100:.1f}cm")
    print(f"Position error: {error*100:.1f}cm")

def ik_move(robot, camera, target: np.ndarray, max_steps: int = 50):
    """Move to target EE position using iterative IK."""
    prev_error = float('inf')
    stuck_count = 0

    for step in range(max_steps):
        update_camera_preview(camera)

        obs = robot.get_observation()
        current_deg = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES])
        current_rad = np.deg2rad(current_deg)

        robot._sync_mujoco(current_rad)
        current_ee = robot._get_ee_position()
        error = np.linalg.norm(target - current_ee)

        if error < 0.015:
            print(f"  Converged at step {step}, error={error*100:.1f}cm")
            break

        if abs(error - prev_error) < 0.0005:
            stuck_count += 1
            if stuck_count >= 3:
                print(f"  Stuck at step {step}, error={error*100:.1f}cm")
                break
        else:
            stuck_count = 0
        prev_error = error

        target_rad = robot._compute_ik(target, current_rad)
        target_deg = np.rad2deg(target_rad)
        target_deg = clamp_degrees(target_deg)

        action_dict = {f"{name}.pos": target_deg[i] for i, name in enumerate(JOINT_NAMES)}
        action_dict["gripper.pos"] = 50.0
        robot.send_action(action_dict)
        busy_wait(0.1)

def main():
    global _robot, _camera

    # Robot config
    config = SO101FollowerEndEffectorConfig(
        id="ggando_so101_follower",
        port="/dev/ttyACM0",
        calibration_dir=Path("/home/gota/.cache/huggingface/lerobot/calibration/robots/so101_follower"),
        mujoco_model_path=Path("/home/gota/ggando/ml/pick-101/models/so101/lift_cube.xml"),
        end_effector_site="gripperframe",
        max_gripper_pos=50.0,
    )

    # Camera config (matching training setup)
    camera_config = OpenCVCameraConfig(
        fps=30,
        width=640,
        height=480,
        index_or_path="/dev/video0",
        color_mode="rgb",
        rotation=0,
    )

    print("Connecting robot...")
    _robot = SO101FollowerEndEffector(config)
    _robot.connect()

    print("Connecting camera...")
    _camera = OpenCVCamera(camera_config)
    _camera.connect()

    print("\n" + "="*50)
    print("IK Height Test")
    print("="*50)
    print("Testing heights: 3cm, 5cm, 7cm, 10cm")
    print("Press Ctrl+C anytime to return to home and exit")
    print("="*50)

    try:
        for z in heights:
            ik_reset_motion(_robot, _camera, z)

            # Keep updating camera while waiting for input
            print("\nPress Enter to continue to next height (or Ctrl+C to exit)...")
            while True:
                update_camera_preview(_camera)
                # Check for Enter key in OpenCV window
                key = cv2.waitKey(100)
                if key == 13 or key == 10:  # Enter key
                    break
                # Also check stdin for Enter
                import select
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()
                    break

        print("\n" + "="*50)
        print("All heights tested!")
        print("="*50)

    finally:
        cleanup()
        print("\nPress Enter to exit (torque will be disabled)...")
        try:
            input()
        except EOFError:
            pass

if __name__ == "__main__":
    main()
