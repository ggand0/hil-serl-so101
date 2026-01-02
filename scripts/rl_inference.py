#!/usr/bin/env python3
"""Real robot inference with trained RL policy.

This script runs a trained DrQ-v2 policy on the real SO-101 robot.

Usage:
    python scripts/rl_inference.py --checkpoint /path/to/best_snapshot.pt

Requirements:
    - Trained checkpoint from pick-101
    - SO-101 robot connected via USB
    - Wrist camera connected

Architecture:
    Camera → Preprocess → Frame Stack ─┐
                                       ├──→ Policy → Cartesian Action → IK → Joint Commands → Robot
    Robot State → Build low_dim_state ─┘

IMPORTANT: The sim uses Cartesian (end-effector) control with IK.
           The real robot uses joint position control via LeRobot.
           We need to implement IK for the real robot or use a different approach.

Current status: SCAFFOLD - needs IK integration for real deployment.
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

# Add pick-101 paths for robobase and training modules
pick101_root = Path("/home/gota/ggando/ml/pick-101")
sys.path.insert(0, str(pick101_root))
sys.path.insert(0, str(pick101_root / "external" / "robobase"))


class RealRobotInterface:
    """Interface to real SO-101 robot via LeRobot.

    This is a placeholder that needs to be implemented with actual LeRobot integration.
    """

    def __init__(
        self,
        robot_port: str = "/dev/ttyACM0",
        robot_id: str = "ggando_so101_follower",
    ):
        self.robot_port = robot_port
        self.robot_id = robot_id
        self._robot = None

        # Joint limits for SO-101 (in radians, approximate)
        self.joint_limits = {
            "lower": np.array([-2.0, -2.0, -2.0, -1.57, -1.57, -1.57]),
            "upper": np.array([2.0, 2.0, 2.0, 1.57, 1.57, 1.57]),
        }

        # Current state
        self._joint_pos = np.zeros(6)
        self._joint_vel = np.zeros(6)
        self._gripper_state = 0.0

    def connect(self) -> bool:
        """Connect to robot.

        Returns:
            True if connected successfully.
        """
        try:
            # TODO: Implement LeRobot connection
            # from lerobot.common.robot_devices.robots.so101_follower import SO101Follower
            # self._robot = SO101Follower(port=self.robot_port, id=self.robot_id)
            # self._robot.connect()

            print(f"[MOCK] Would connect to robot at {self.robot_port}")
            return True

        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False

    def disconnect(self):
        """Disconnect from robot."""
        if self._robot is not None:
            # self._robot.disconnect()
            pass
        print("[MOCK] Robot disconnected")

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions.

        Returns:
            Joint positions (6,) in radians.
        """
        # TODO: Read from actual robot
        # return self._robot.get_joint_positions()
        return self._joint_pos.copy()

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities.

        Returns:
            Joint velocities (6,) in rad/s.
        """
        # TODO: Compute from position history or read from robot
        return self._joint_vel.copy()

    def get_gripper_state(self) -> float:
        """Get gripper state.

        Returns:
            Gripper position (0=closed, 1=open).
        """
        # TODO: Read from actual robot
        return self._gripper_state

    def set_joint_positions(self, positions: np.ndarray):
        """Set target joint positions.

        Args:
            positions: Target joint positions (6,) in radians.
        """
        # Clamp to joint limits
        positions = np.clip(positions, self.joint_limits["lower"], self.joint_limits["upper"])

        # TODO: Send to actual robot
        # self._robot.set_joint_positions(positions)

        self._joint_pos = positions.copy()
        print(f"[MOCK] Set joints: {positions}")

    def set_gripper(self, position: float):
        """Set gripper position.

        Args:
            position: Gripper position (0=closed, 1=open).
        """
        position = np.clip(position, 0.0, 1.0)

        # TODO: Send to actual robot
        # self._robot.set_gripper(position)

        self._gripper_state = position
        print(f"[MOCK] Set gripper: {position:.2f}")


class CartesianToJointConverter:
    """Converts Cartesian actions to joint commands.

    The RL policy outputs:
    - delta XYZ (3): End-effector position change, scaled by 0.02 (2cm/step)
    - gripper (1): Gripper open/close in [-1, 1]

    We need to convert these to joint position commands for the real robot.

    Options:
    1. Numerical IK (slow but accurate)
    2. Analytical IK for SO-101 (fast, requires derivation)
    3. Jacobian-based velocity control

    Current implementation: Placeholder using Jacobian approximation.
    """

    def __init__(self, action_scale: float = 0.02):
        """Initialize converter.

        Args:
            action_scale: Scale factor for delta XYZ (meters per action unit).
        """
        self.action_scale = action_scale

        # Approximate Jacobian for SO-101 at typical operating pose
        # This is a placeholder - real implementation needs proper kinematics
        self._jacobian = np.eye(6)[:3, :]  # 3x6 matrix

    def convert(
        self,
        action: np.ndarray,
        current_joint_pos: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Convert Cartesian action to joint command.

        Args:
            action: Policy output (4,) = [delta_x, delta_y, delta_z, gripper].
            current_joint_pos: Current joint positions (6,).

        Returns:
            Tuple of (target_joint_pos (6,), gripper_position (0-1)).
        """
        # Extract action components
        delta_xyz = action[:3] * self.action_scale  # Scale to meters
        gripper_action = action[3]  # [-1, 1]

        # Convert gripper action to position
        # -1 = closed, 1 = open → 0 = closed, 1 = open
        gripper_pos = (gripper_action + 1.0) / 2.0

        # TODO: Implement proper IK
        # For now, use simple Jacobian pseudoinverse (very approximate)
        # delta_joints = J_pinv @ delta_xyz
        j_pinv = np.linalg.pinv(self._jacobian)
        delta_joints = j_pinv @ delta_xyz

        target_joint_pos = current_joint_pos + delta_joints

        return target_joint_pos, gripper_pos


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
        help="Run without sending commands to robot",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for inference",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SO-101 RL Inference")
    print("=" * 60)

    # Initialize components
    print("\n[1/4] Loading policy...")
    policy = PolicyRunner(args.checkpoint, device=args.device)
    if not policy.load():
        print("Failed to load policy. Exiting.")
        return

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
            print("No camera available, using mock observations.")
            use_mock_camera = True
            camera = None
        else:
            camera.warm_up()
    else:
        print("[DRY RUN] Using mock camera observations")

    print("\n[3/4] Connecting to robot...")
    robot = RealRobotInterface(robot_port=args.robot_port)
    if not args.dry_run and not robot.connect():
        print("Failed to connect to robot. Exiting.")
        camera.close()
        return

    print("\n[4/4] Initializing state builder and action converter...")
    state_builder = LowDimStateBuilder(include_cube_pos=False)  # No cube detection
    action_converter = CartesianToJointConverter(action_scale=0.02)

    # Frame buffer for stacking low_dim_state
    state_buffer = deque(maxlen=policy.frame_stack)

    print("\n" + "=" * 60)
    print("Ready to run. Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        for episode in range(args.num_episodes):
            print(f"\n--- Episode {episode + 1}/{args.num_episodes} ---")

            # Reset
            state_buffer.clear()
            if camera is not None:
                camera.reset()
                if not camera.fill_buffer():
                    print("Failed to fill camera buffer")
                    continue

            # End-effector state (fixed approximation - TODO: compute from FK)
            ee_pos = np.array([0.25, 0.0, 0.05])  # Approximate EE position
            ee_euler = np.array([0.0, np.pi / 2, 0.0])  # Top-down orientation

            # Fill state buffer with initial states
            for _ in range(policy.frame_stack):
                joint_pos = robot.get_joint_positions()
                joint_vel = robot.get_joint_velocities()

                state = state_builder.build(
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    gripper_pos=ee_pos,
                    gripper_euler=ee_euler,
                )
                state_buffer.append(state)

            # Run episode
            for step in range(args.episode_length):
                # Get observations
                if use_mock_camera:
                    # Mock RGB observation: random noise (frame_stack, 3, 84, 84)
                    rgb_obs = np.random.randint(0, 256, (policy.frame_stack, 3, 84, 84), dtype=np.uint8)
                else:
                    rgb_obs = camera.get_stacked_observation()
                    if rgb_obs is None:
                        print("Camera frame error")
                        break

                # Stack low_dim states
                low_dim_obs = np.stack(list(state_buffer), axis=0).astype(np.float32)

                # Get action from policy
                action = policy.get_action(rgb_obs, low_dim_obs)

                # Convert to joint commands
                current_joints = robot.get_joint_positions()
                target_joints, gripper_cmd = action_converter.convert(action, current_joints)

                # Execute action
                if not args.dry_run:
                    robot.set_joint_positions(target_joints)
                    robot.set_gripper(gripper_cmd)

                # Update state for next step
                joint_pos = robot.get_joint_positions()
                joint_vel = robot.get_joint_velocities()
                # TODO: Update ee_pos from FK (currently using fixed approximate)
                state = state_builder.build(
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    gripper_pos=ee_pos,
                    gripper_euler=ee_euler,
                )
                state_buffer.append(state)

                # Status
                if step % 20 == 0:
                    print(f"  Step {step}: action={action[:3]} gripper={action[3]:.2f}")

                # Control rate
                time.sleep(0.05)  # 20 Hz

            print(f"Episode {episode + 1} complete")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("\nCleaning up...")
        if camera is not None:
            camera.close()
        if not args.dry_run:
            robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
