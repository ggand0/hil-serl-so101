#!/usr/bin/env python
"""Kinematic verification: compare MuJoCo FK vs real robot EE positions.

This script moves the robot to several joint configurations and compares
the MuJoCo-computed EE position with manual measurements or markers.

Usage:
    cd /home/gota/ggando/ml/lerobot
    uv run python /home/gota/ggando/ml/so101-playground/scripts/verify_kinematics.py
    uv run python /home/gota/ggando/ml/so101-playground/scripts/verify_kinematics.py --no-interactive
"""

import argparse
import json
import logging
import numpy as np
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

import mujoco


# Joint names in order
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Test configurations (in degrees)
# These are designed to cover different parts of the workspace
TEST_CONFIGS = [
    {
        "name": "Home (all zeros)",
        "joints_deg": [0, 0, 0, 0, 0],
        "description": "Extended forward position"
    },
    {
        "name": "Top-down ready",
        "joints_deg": [0, 0, 0, 90, -90],
        "description": "Wrist oriented for top-down grasp"
    },
    {
        "name": "Bent elbow",
        "joints_deg": [0, -45, 45, 90, -90],
        "description": "Elbow bent, wrist top-down"
    },
    {
        "name": "Side reach",
        "joints_deg": [45, -30, 30, 90, -90],
        "description": "Reaching to the side"
    },
    {
        "name": "Low position",
        "joints_deg": [0, -60, 90, 90, -90],
        "description": "Low reach position"
    },
]


class KinematicVerifier:
    def __init__(self, mujoco_model_path: str, ee_site: str = "gripperframe"):
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.ee_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_site
        )
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{ee_site}' not found in MuJoCo model")

        print(f"Loaded MuJoCo model: {mujoco_model_path}")
        print(f"EE site: {ee_site} (id={self.ee_site_id})")
        print(f"Model has {self.mj_model.nq} qpos, {self.mj_model.nv} dof")

    def compute_fk(self, joints_rad: np.ndarray) -> np.ndarray:
        """Compute forward kinematics to get EE position."""
        self.mj_data.qpos[:5] = joints_rad[:5]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.mj_data.site_xpos[self.ee_site_id].copy()

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Get joint limits from model."""
        lower = np.array([self.mj_model.jnt_range[i, 0] for i in range(5)])
        upper = np.array([self.mj_model.jnt_range[i, 1] for i in range(5)])
        return lower, upper


def main():
    parser = argparse.ArgumentParser(description="Kinematic verification test")
    parser.add_argument("--no-interactive", action="store_true", help="Run without user prompts")
    parser.add_argument("--current-pose", action="store_true", help="Just read current pose and compute FK (no movement)")
    args = parser.parse_args()

    interactive = not args.no_interactive

    from lerobot.motors import MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor

    # Initialize MuJoCo verifier
    mujoco_path = "/home/gota/ggando/ml/pick-101/models/so101/lift_cube_calibration.xml"
    verifier = KinematicVerifier(mujoco_path)

    # Reference point: front_base_marker (front edge of robot base, on table)
    FRONT_BASE_REF = np.array([0.08, 0.0, 0.015])

    # Get joint limits
    lower, upper = verifier.get_joint_limits()
    print(f"\nJoint limits (rad):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {name}: [{lower[i]:.3f}, {upper[i]:.3f}] = [{np.rad2deg(lower[i]):.1f}°, {np.rad2deg(upper[i]):.1f}°]")

    # Load calibration
    from lerobot.motors import MotorCalibration

    calib_path = Path("/home/gota/.cache/huggingface/lerobot/calibration/robots/so101_follower/ggando_so101_follower.json")
    with open(calib_path) as f:
        calib_dict = json.load(f)

    # Convert to MotorCalibration objects
    calibration = {
        name: MotorCalibration(
            id=data["id"],
            drive_mode=data["drive_mode"],
            homing_offset=data["homing_offset"],
            range_min=data["range_min"],
            range_max=data["range_max"],
        )
        for name, data in calib_dict.items()
    }

    # Connect to robot using degrees mode
    bus = FeetechMotorsBus(
        port="/dev/ttyACM0",
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )
    bus.connect()
    print("\nConnected to robot")

    # Current pose mode - just read and compute FK, no movement
    if args.current_pose:
        print("\n" + "=" * 60)
        print("CURRENT POSE FK")
        print("=" * 60)

        pos_dict = bus.sync_read("Present_Position")
        current_deg = np.array([pos_dict[name] for name in JOINT_NAMES])
        current_rad = np.deg2rad(current_deg)

        ee_pos = verifier.compute_fk(current_rad)

        # Delta from front_base_marker reference point
        delta = ee_pos - FRONT_BASE_REF

        print(f"\nCurrent joints (deg): {current_deg}")
        print(f"\nReference point (front_base_marker): [{FRONT_BASE_REF[0]*100:.1f}, {FRONT_BASE_REF[1]*100:.1f}, {FRONT_BASE_REF[2]*100:.1f}] cm")
        print(f"Current EE position:                  [{ee_pos[0]*100:.2f}, {ee_pos[1]*100:.2f}, {ee_pos[2]*100:.2f}] cm")
        print(f"\nDistance from front_base_marker:")
        print(f"  X (forward):  {delta[0]*100:+.2f} cm")
        print(f"  Y (left):     {delta[1]*100:+.2f} cm")
        print(f"  Z (up):       {delta[2]*100:+.2f} cm")

        bus.disconnect()
        print("\nDisconnected (no movement performed)")
        return

    # Results storage
    results = []

    print("\n" + "=" * 60)
    print("KINEMATIC VERIFICATION TEST")
    print("=" * 60)
    print("\nThis test will move the robot to several configurations.")
    print("For each configuration, note the physical EE position if possible.")
    if interactive:
        print("Press Enter to start, Ctrl+C to abort.\n")
        input()
    else:
        print("Running in non-interactive mode...\n")

    try:
        for config in TEST_CONFIGS:
            print(f"\n--- {config['name']} ---")
            print(f"Description: {config['description']}")

            joints_deg = np.array(config["joints_deg"], dtype=np.float64)
            joints_rad = np.deg2rad(joints_deg)

            # Compute MuJoCo FK
            mujoco_ee = verifier.compute_fk(joints_rad)

            print(f"Target joints (deg): {joints_deg}")
            print(f"MuJoCo EE position: [{mujoco_ee[0]:.4f}, {mujoco_ee[1]:.4f}, {mujoco_ee[2]:.4f}] m")

            # Move robot
            print("Moving robot...")
            action_dict = {name: joints_deg[i] for i, name in enumerate(JOINT_NAMES)}
            action_dict["gripper"] = 50.0  # Open gripper
            bus.sync_write("Goal_Position", action_dict)

            # Wait for motion
            time.sleep(2.0)

            # Read actual position
            pos_dict = bus.sync_read("Present_Position")
            actual_deg = np.array([pos_dict[name] for name in JOINT_NAMES])
            actual_rad = np.deg2rad(actual_deg)

            # Compute FK with actual positions
            actual_mujoco_ee = verifier.compute_fk(actual_rad)

            joint_error_deg = actual_deg - joints_deg

            print(f"Actual joints (deg): {actual_deg}")
            print(f"Joint error (deg): {joint_error_deg}")
            print(f"FK with actual joints: [{actual_mujoco_ee[0]:.4f}, {actual_mujoco_ee[1]:.4f}, {actual_mujoco_ee[2]:.4f}] m")

            results.append({
                "name": config["name"],
                "target_joints_deg": joints_deg.tolist(),
                "actual_joints_deg": actual_deg.tolist(),
                "joint_error_deg": joint_error_deg.tolist(),
                "mujoco_ee_target": mujoco_ee.tolist(),
                "mujoco_ee_actual": actual_mujoco_ee.tolist(),
            })

            if interactive:
                print("\nPress Enter to continue to next configuration...")
                input()
            else:
                time.sleep(0.5)  # Brief pause between configs

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print("\nJoint tracking errors (deg):")
        for r in results:
            err = np.array(r["joint_error_deg"])
            print(f"  {r['name']}: max={np.abs(err).max():.2f}°, mean={np.abs(err).mean():.2f}°")

        print("\nMuJoCo EE positions (with actual joints):")
        for r in results:
            ee = r["mujoco_ee_actual"]
            print(f"  {r['name']}: [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}] m")

        # Save results
        results_path = Path("/home/gota/ggando/ml/so101-playground/kinematic_verification_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        print("\n" + "=" * 60)
        print("MANUAL VERIFICATION INSTRUCTIONS")
        print("=" * 60)
        print("""
To verify kinematic accuracy:
1. Place a ruler or measuring tape at the robot base
2. For each configuration, measure the physical EE position
3. Compare with MuJoCo FK values above
4. Acceptable error: < 1cm for manipulation tasks

If errors are large (> 2cm), possible causes:
- URDF joint origins incorrect
- Calibration homing offset wrong
- Joint axis directions inverted
""")

    except KeyboardInterrupt:
        print("\nAborted by user")
    finally:
        # First tuck the arm safely (bend elbow/wrist) before going to home
        print("\nTucking arm safely...")
        tuck_joints = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 90,
                       "wrist_flex": 90, "wrist_roll": 0, "gripper": 50}
        bus.sync_write("Goal_Position", tuck_joints)
        time.sleep(2.0)

        # Now go to home position
        print("Returning to home position...")
        home_joints = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
                       "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}
        bus.sync_write("Goal_Position", home_joints)
        time.sleep(2.0)

        bus.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
