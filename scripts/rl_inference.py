#!/usr/bin/env python3
"""Real robot inference with trained RL policy.

Runs a trained DrQ-v2 policy on the real SO-101 robot.

Usage:
    # Dry run (mock robot and camera)
    uv run python scripts/rl_inference.py --checkpoint /path/to/snapshot.pt --dry_run

    # Real robot
    uv run python scripts/rl_inference.py --checkpoint /path/to/snapshot.pt

Architecture:
    Camera → Preprocess → Frame Stack ─┐
                                       ├─→ Policy → Cartesian Action → IK → Joint Commands → Robot
    Robot State → FK → low_dim_state ──┘
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np

# Add project root to path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import local deploy modules first (before adding pick-101 which also has 'src')
from src.deploy.camera import CameraPreprocessor
from src.deploy.policy import PolicyRunner, LowDimStateBuilder
from src.deploy.robot import SO101Robot, MockSO101Robot
from src.deploy.controllers import IKController

# Add pick-101 paths for robobase and training modules
pick101_root = Path("/home/gota/ggando/ml/pick-101")
sys.path.insert(0, str(pick101_root))
sys.path.insert(0, str(pick101_root / "external" / "robobase"))


def main():
    parser = argparse.ArgumentParser(description="Run RL policy on real SO-101 robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera device index",
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Robot serial port",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=200,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without real robot/camera (mock mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for inference",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=0.02,
        help="Action scale (meters per unit, default 2cm)",
    )
    parser.add_argument(
        "--control_hz",
        type=float,
        default=10.0,
        help="Control frequency in Hz",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SO-101 RL Inference")
    print("=" * 60)

    # === 1. Load Policy ===
    print("\n[1/4] Loading policy...")
    policy = PolicyRunner(args.checkpoint, device=args.device)
    if not policy.load():
        print("Failed to load policy. Exiting.")
        return
    print(f"  Frame stack: {policy.frame_stack}")

    # === 2. Initialize Camera ===
    print("\n[2/4] Initializing camera...")
    camera = None
    use_mock_camera = args.dry_run

    if not use_mock_camera:
        camera = CameraPreprocessor(
            target_size=(84, 84),
            frame_stack=policy.frame_stack,
            camera_index=args.camera_index,
        )
        if not camera.open():
            print("  No camera available, using mock observations.")
            use_mock_camera = True
            camera = None
        else:
            camera.warm_up()
            print(f"  Camera opened at index {args.camera_index}")
    else:
        print("  [DRY RUN] Using mock camera observations")

    # === 3. Initialize Robot ===
    print("\n[3/4] Initializing robot...")
    if args.dry_run:
        robot = MockSO101Robot(port=args.robot_port)
        robot.connect()
    else:
        robot = SO101Robot(port=args.robot_port)
        if not robot.connect():
            print("Failed to connect to robot. Exiting.")
            if camera is not None:
                camera.close()
            return

    # === 4. Initialize IK Controller ===
    print("\n[4/4] Initializing IK controller...")
    try:
        ik = IKController()
        print(f"  IK controller ready (damping={ik.damping})")
    except Exception as e:
        print(f"  Failed to initialize IK: {e}")
        if camera is not None:
            camera.close()
        robot.disconnect()
        return

    # State builder (no cube position in real deployment)
    state_builder = LowDimStateBuilder(include_cube_pos=False)

    # Frame buffer for stacking low_dim_state
    state_buffer = deque(maxlen=policy.frame_stack)

    # Control timing
    control_dt = 1.0 / args.control_hz

    print("\n" + "=" * 60)
    print("Ready to run. Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        for episode in range(args.num_episodes):
            print(f"\n--- Episode {episode + 1}/{args.num_episodes} ---")

            # Reset buffers
            state_buffer.clear()
            if camera is not None:
                camera.reset()
                if not camera.fill_buffer():
                    print("  Failed to fill camera buffer, skipping episode")
                    continue

            # Get initial robot state and compute FK
            joint_pos_rad = robot.get_joint_positions_radians()
            joint_vel = robot.get_joint_velocities()
            gripper_state = robot.get_gripper_position()

            # Use IK controller for FK (sync joints, read EE pose)
            ik.sync_joint_positions(joint_pos_rad)
            ee_pos = ik.get_ee_position()
            ee_euler = ik.get_ee_euler()

            print(f"  Initial EE position: {ee_pos}")
            print(f"  Initial joints (rad): {joint_pos_rad}")

            # Fill state buffer
            for _ in range(policy.frame_stack):
                state = state_builder.build(
                    joint_pos=joint_pos_rad,
                    joint_vel=joint_vel,
                    gripper_pos=ee_pos,
                    gripper_euler=ee_euler,
                    gripper_state=gripper_state,
                )
                state_buffer.append(state)

            # Episode loop
            for step in range(args.episode_length):
                step_start = time.time()

                # Get RGB observation
                if use_mock_camera:
                    rgb_obs = np.random.randint(
                        0, 256, (policy.frame_stack, 3, 84, 84), dtype=np.uint8
                    )
                else:
                    rgb_obs = camera.get_stacked_observation()
                    if rgb_obs is None:
                        print("  Camera frame error, ending episode")
                        break

                # Stack low_dim states
                low_dim_obs = np.stack(list(state_buffer), axis=0).astype(np.float32)

                # Get action from policy
                action = policy.get_action(rgb_obs, low_dim_obs)

                # Parse action
                delta_xyz = action[:3]  # In [-1, 1], will be scaled
                gripper_action = action[3]  # -1 = closed, 1 = open

                # Use IK to convert Cartesian action to joint targets
                current_joints = robot.get_joint_positions_radians()
                target_joints = ik.cartesian_to_joints(
                    delta_xyz,
                    current_joints,
                    action_scale=args.action_scale,
                    locked_joints=[4],  # Lock wrist_roll for stability
                )

                # Send action to robot
                if not args.dry_run:
                    robot.send_action(target_joints, gripper_action)

                # Update state for next step
                joint_pos_rad = robot.get_joint_positions_radians()
                joint_vel = robot.get_joint_velocities()
                gripper_state = robot.get_gripper_position()

                # FK for EE state
                ik.sync_joint_positions(joint_pos_rad)
                ee_pos = ik.get_ee_position()
                ee_euler = ik.get_ee_euler()

                state = state_builder.build(
                    joint_pos=joint_pos_rad,
                    joint_vel=joint_vel,
                    gripper_pos=ee_pos,
                    gripper_euler=ee_euler,
                    gripper_state=gripper_state,
                )
                state_buffer.append(state)

                # Status
                if step % 20 == 0:
                    print(
                        f"  Step {step}: delta={delta_xyz} "
                        f"gripper={gripper_action:.2f} ee_z={ee_pos[2]:.3f}"
                    )

                # Control rate
                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"  Episode {episode + 1} complete")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("\nCleaning up...")
        if camera is not None:
            camera.close()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
