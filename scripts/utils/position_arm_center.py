#!/usr/bin/env python3
"""Position arm at center [0.25, 0.0, 0.07] and show camera preview."""
import sys
sys.path.insert(0, "/home/gota/ggando/ml/lerobot/src")

import cv2
import time
import numpy as np
from pathlib import Path
from lerobot.robots.so101_follower import SO101FollowerEndEffectorConfig, SO101FollowerEndEffector

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TARGET = np.array([0.25, 0.0, 0.07])

def clamp_degrees(joints_deg):
    limits = [(-180, 180), (-90, 90), (-90, 90), (-90, 90), (-180, 180)]
    return np.array([np.clip(joints_deg[i], lo, hi) for i, (lo, hi) in enumerate(limits)])

config = SO101FollowerEndEffectorConfig(
    id="ggando_so101_follower",
    port="/dev/ttyACM0",
    calibration_dir=Path("/home/gota/.cache/huggingface/lerobot/calibration/robots/so101_follower"),
    mujoco_model_path=Path("/home/gota/ggando/ml/pick-101/models/so101/lift_cube.xml"),
)
robot = SO101FollowerEndEffector(config)
robot.connect()

# Step 1: Safe position
print("Step 1: Safe position")
robot.send_action({f"{n}.pos": 0.0 for n in JOINT_NAMES} | {"gripper.pos": 50.0})
time.sleep(1.5)

# Step 2: Wrist down
print("Step 2: Wrist orientation")
obs = robot.get_observation()
current = np.array([obs[f"{n}.pos"] for n in JOINT_NAMES])
current[3], current[4] = 90.0, 90.0
robot.send_action({f"{n}.pos": current[i] for i, n in enumerate(JOINT_NAMES)} | {"gripper.pos": 50.0})
time.sleep(1.0)

# Step 3-4: IK move (above then target)
for label, target in [("Step 3: Above", TARGET + [0, 0, 0.07]), ("Step 4: Lower", TARGET)]:
    print(f"{label}: {target}")
    for _ in range(50):
        obs = robot.get_observation()
        current_deg = np.array([obs[f"{n}.pos"] for n in JOINT_NAMES])
        current_rad = np.deg2rad(current_deg)
        robot._sync_mujoco(current_rad)
        if np.linalg.norm(target - robot._get_ee_position()) < 0.015:
            break
        target_rad = robot._compute_ik(target, current_rad)
        target_deg = clamp_degrees(np.rad2deg(target_rad))
        robot.send_action({f"{n}.pos": target_deg[i] for i, n in enumerate(JOINT_NAMES)} | {"gripper.pos": 50.0})
        time.sleep(0.1)

print("\nArm at center. Opening camera preview...")
cap = cv2.VideoCapture("/dev/video0")
print("Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
robot.disconnect()
