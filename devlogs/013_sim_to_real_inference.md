# Devlog 013: Sim-to-Real RL Inference

## Summary

Set up real robot inference with trained DrQ-v2 policy (v17 from pick-101). Key work included matching the training reset position, camera calibration, and IK-based positioning.

## Training Configuration

The v17 policy was trained with:
- `curriculum_stage: 3` - gripper starts above cube
- Wrist joints locked at π/2 (top-down orientation)
- Cube at ~[0.25, 0.0, 0.015]
- Gripper positioned 3cm above cube

## Reset Position Matching

### Problem

Initial inference attempts failed because:
1. Robot started from home position (extended forward at [0.39, 0, 0.23])
2. Training starts with gripper positioned above cube (~[0.25, -0.015, 0.05])
3. Policy was confused by the position mismatch

### Solution

Match the training reset sequence:
1. Start from safe extended position (all joints = 0)
2. Set wrist joints to π/2 for top-down orientation
3. Use IK with locked wrist to move above cube position

```python
# Training initial position constants
FINGER_WIDTH_OFFSET = -0.015  # Static finger offset
GRASP_Z_OFFSET = 0.005
HEIGHT_OFFSET = 0.03  # 3cm above grasp height
CUBE_Z = 0.015  # Cube on table

# Target position above cube
initial_target = np.array([
    cube_x,
    cube_y + FINGER_WIDTH_OFFSET,
    CUBE_Z + GRASP_Z_OFFSET + HEIGHT_OFFSET  # ~0.05m
])
```

## IK Controller

Uses MuJoCo as kinematics solver for damped least-squares IK:
- Sync real robot joint positions to MuJoCo model
- Compute Jacobian and IK for Cartesian targets
- Lock wrist joints during IK for stable orientation

Key insight: Training uses `locked_joints=[3, 4]` during reset, so real robot must match.

## Tools Created

### ik_reset_position.py

Calibration tool to position robot for marking cube location:

```bash
# Move to training initial position
uv run python scripts/ik_reset_position.py

# Also lower to grasp height
uv run python scripts/ik_reset_position.py --lower

# Record current robot position
uv run python scripts/ik_reset_position.py --record_only

# Adjust cube position
uv run python scripts/ik_reset_position.py --cube_x 0.28 --cube_y -0.02
```

Features:
- Safe sequence: extended → wrist setup → IK to target
- Safe return: lift up → rest position
- Record mode: read current joints for calibration

### Rest Position

Recorded user's preferred rest position:
```python
REST_JOINTS = np.array([-0.247, -1.8132, 1.6812, 1.2187, -2.9821])
```

## Camera Calibration (from devlog 012)

- MuJoCo `cam_fovy` is VERTICAL FOV, not horizontal
- Real camera: 103° horizontal → 86° vertical
- Camera pitch: -26° (7cm offset, 13cm forward)
- Center crop: 640x480 → 480x480 → 84x84

## Files Modified

- `scripts/rl_inference.py` - Added proper reset sequence with wrist lock
- `scripts/ik_reset_position.py` - New calibration tool
- `src/deploy/camera.py` - Center crop for square aspect ratio
- `scripts/record_camera.py` - Debug camera with center crop

## Next Steps

1. Run inference with matched reset position
2. Tune action_scale for real robot (may differ from sim)
3. Add safety limits for joint velocities
4. Record and analyze failure modes
