"""Real robot interface using LeRobot for SO-101.

LeRobot uses normalized positions:
- Arm joints: -100 to 100 range
- Gripper: 0 to 100 range

MuJoCo/Policy uses radians:
- Arm joints: approximately -π to π
- Gripper: -1 to 1 (policy output)

This module handles the conversion between these formats.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


# Motor names in LeRobot order
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Load calibration from LeRobot cache
CALIBRATION_PATH = Path("/home/gota/.cache/huggingface/lerobot/calibration/robots/so101_follower/ggando_so101_follower.json")

def _load_calibration():
    with open(CALIBRATION_PATH) as f:
        return json.load(f)

CALIBRATION = _load_calibration()

# Feetech servo: 4096 encoder positions per revolution
ENCODER_PER_RAD = 4096 / (2 * np.pi)  # ~652 encoder units per radian


def normalized_to_radians(normalized: float, joint_name: str) -> float:
    """Convert LeRobot normalized position (-100 to 100) to radians."""
    cal = CALIBRATION[joint_name]
    range_min = cal["range_min"]
    range_max = cal["range_max"]

    # Normalized to encoder position
    encoder = (normalized + 100) / 200 * (range_max - range_min) + range_min

    # Encoder to radians (mid-range = 0 radians)
    mid_encoder = (range_min + range_max) / 2
    radians = (encoder - mid_encoder) / ENCODER_PER_RAD

    return radians


def radians_to_normalized(radians: float, joint_name: str) -> float:
    """Convert radians to LeRobot normalized position (-100 to 100)."""
    cal = CALIBRATION[joint_name]
    range_min = cal["range_min"]
    range_max = cal["range_max"]

    # Radians to encoder (mid-range = 0 radians)
    mid_encoder = (range_min + range_max) / 2
    encoder = radians * ENCODER_PER_RAD + mid_encoder

    # Clamp to range
    encoder = np.clip(encoder, range_min, range_max)

    # Encoder to normalized
    normalized = (encoder - range_min) / (range_max - range_min) * 200 - 100

    return normalized


def gripper_policy_to_lerobot(policy_output: float) -> float:
    """Convert policy gripper output (-1 to 1) to LeRobot (0 to 100).

    Policy: -1 = closed, 1 = open
    LeRobot: 0 = closed, 100 = open
    """
    # -1 -> 0 (closed), 1 -> 100 (open)
    return (policy_output + 1) / 2 * 100


def gripper_lerobot_to_policy(lerobot_pos: float) -> float:
    """Convert LeRobot gripper (0 to 100) to policy range (-1 to 1)."""
    # 0 -> -1 (closed), 100 -> 1 (open)
    return lerobot_pos / 50 - 1


class SO101Robot:
    """Interface to SO-101 robot via LeRobot.

    Handles:
    - Connection to real robot
    - Position format conversion (normalized <-> radians)
    - Joint velocity estimation
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        robot_id: str = "ggando_so101_follower",
    ):
        self.port = port
        self.robot_id = robot_id
        self._robot = None
        self._connected = False

        # State tracking
        self._last_positions: np.ndarray | None = None
        self._last_time: float | None = None

    def connect(self) -> bool:
        """Connect to the real robot."""
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

            config = SO101FollowerConfig(
                port=self.port,
                id=self.robot_id,
                disable_torque_on_disconnect=True,
                use_degrees=False,  # Use normalized -100 to 100
            )

            self._robot = SO101Follower(config)
            self._robot.connect(calibrate=False)  # Assume already calibrated
            self._connected = True
            print(f"Connected to SO-101 at {self.port}")
            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from robot."""
        if self._robot is not None:
            self._robot.disconnect()
            self._connected = False
            print("Disconnected from SO-101")

    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> dict[str, Any]:
        """Get raw observation from robot."""
        if not self._connected:
            raise RuntimeError("Robot not connected")
        return self._robot.get_observation()

    def get_joint_positions_normalized(self) -> np.ndarray:
        """Get joint positions in LeRobot normalized format (-100 to 100)."""
        obs = self.get_observation()
        return np.array([obs[f"{name}.pos"] for name in MOTOR_NAMES])

    def get_joint_positions_radians(self) -> np.ndarray:
        """Get joint positions in radians (for MuJoCo/policy)."""
        normalized = self.get_joint_positions_normalized()
        radians = np.array([
            normalized_to_radians(normalized[i], MOTOR_NAMES[i])
            for i in range(len(MOTOR_NAMES))
        ])
        return radians

    def get_joint_velocities(self) -> np.ndarray:
        """Estimate joint velocities from position changes.

        Returns velocities in rad/s (approximate).
        """
        import time

        current_pos = self.get_joint_positions_radians()
        current_time = time.time()

        if self._last_positions is None or self._last_time is None:
            self._last_positions = current_pos
            self._last_time = current_time
            return np.zeros(5)

        dt = current_time - self._last_time
        if dt < 0.001:  # Avoid division by zero
            return np.zeros(5)

        velocities = (current_pos - self._last_positions) / dt

        self._last_positions = current_pos
        self._last_time = current_time

        return velocities

    def get_gripper_position(self) -> float:
        """Get gripper position in policy format (-1 to 1)."""
        obs = self.get_observation()
        lerobot_pos = obs["gripper.pos"]
        return gripper_lerobot_to_policy(lerobot_pos)

    def send_joint_positions_radians(self, positions: np.ndarray):
        """Send joint positions in radians."""
        if not self._connected:
            raise RuntimeError("Robot not connected")

        # Convert to normalized
        normalized = [
            radians_to_normalized(positions[i], MOTOR_NAMES[i])
            for i in range(len(MOTOR_NAMES))
        ]

        # Build action dict
        action = {f"{name}.pos": normalized[i] for i, name in enumerate(MOTOR_NAMES)}

        self._robot.send_action(action)

    def send_gripper(self, policy_value: float):
        """Send gripper command from policy output (-1 to 1)."""
        if not self._connected:
            raise RuntimeError("Robot not connected")

        lerobot_value = gripper_policy_to_lerobot(policy_value)
        self._robot.send_action({"gripper.pos": lerobot_value})

    def send_action(self, joint_positions_rad: np.ndarray, gripper_policy: float):
        """Send full action (joints + gripper)."""
        if not self._connected:
            raise RuntimeError("Robot not connected")

        # Convert joints
        normalized = [
            radians_to_normalized(joint_positions_rad[i], MOTOR_NAMES[i])
            for i in range(len(MOTOR_NAMES))
        ]

        # Convert gripper
        gripper_normalized = gripper_policy_to_lerobot(gripper_policy)

        # Build action dict
        action = {f"{name}.pos": normalized[i] for i, name in enumerate(MOTOR_NAMES)}
        action["gripper.pos"] = gripper_normalized

        self._robot.send_action(action)


class MockSO101Robot:
    """Mock robot for testing without hardware."""

    def __init__(self, port: str = "/dev/ttyACM0", robot_id: str = "mock"):
        self.port = port
        self.robot_id = robot_id
        self._joint_positions = np.zeros(5)
        self._gripper = 0.0
        self._connected = False

    def connect(self) -> bool:
        print(f"[MOCK] Connected to SO-101 at {self.port}")
        self._connected = True
        return True

    def disconnect(self):
        print("[MOCK] Disconnected")
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_joint_positions_radians(self) -> np.ndarray:
        return self._joint_positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(5)

    def get_gripper_position(self) -> float:
        return self._gripper

    def send_action(self, joint_positions_rad: np.ndarray, gripper_policy: float):
        self._joint_positions = joint_positions_rad.copy()
        self._gripper = gripper_policy
        print(f"[MOCK] Joints: {joint_positions_rad[:3]}... Gripper: {gripper_policy:.2f}")


def test_conversion():
    """Test position conversion functions."""
    print("Testing position conversions...")

    # Test joint conversion
    for name in MOTOR_NAMES:
        low, high = JOINT_RANGES_RAD[name]
        mid = (low + high) / 2

        # Test endpoints
        assert abs(radians_to_normalized(low, name) - (-100)) < 0.1, f"{name} low failed"
        assert abs(radians_to_normalized(high, name) - 100) < 0.1, f"{name} high failed"
        assert abs(radians_to_normalized(mid, name) - 0) < 0.1, f"{name} mid failed"

        # Test roundtrip
        for val in [-100, 0, 100]:
            rad = normalized_to_radians(val, name)
            back = radians_to_normalized(rad, name)
            assert abs(back - val) < 0.1, f"{name} roundtrip failed for {val}"

    # Test gripper conversion
    # Policy: -1 = closed, 1 = open
    # LeRobot: 0 = closed, 100 = open
    assert abs(gripper_policy_to_lerobot(-1) - 0) < 0.1, "Gripper closed failed"
    assert abs(gripper_policy_to_lerobot(1) - 100) < 0.1, "Gripper open failed"
    assert abs(gripper_lerobot_to_policy(0) - (-1)) < 0.1, "Gripper reverse closed failed"
    assert abs(gripper_lerobot_to_policy(100) - 1) < 0.1, "Gripper reverse open failed"

    print("All conversion tests passed!")


if __name__ == "__main__":
    test_conversion()
