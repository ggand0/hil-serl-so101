# Devlog 010: RL Deployment Scaffold

**Date**: January 2, 2026

## Overview

Created the deployment pipeline scaffold for running trained DrQ-v2 policies from `pick-101` on the real SO-101 robot. This is preparation for sim-to-real transfer.

## Files Created

```
so101-playground/
├── src/
│   ├── __init__.py
│   └── deploy/
│       ├── __init__.py
│       ├── camera.py      # Camera capture + preprocessing
│       └── policy.py      # Policy loading + inference
├── scripts/
│   └── rl_inference.py    # Main inference loop
└── configs/
    └── deploy.yaml        # Deployment configuration
```

## Architecture

### Observation Pipeline

```
Real Camera (640x480, 130° FOV)
    ↓
Center crop to match sim FOV (75°)
    ↓
Resize to 84×84
    ↓
RGB → CHW format
    ↓
Frame stack (3 frames)
    ↓
Combine with low_dim_state
    ↓
Policy inference → Action
```

### Action Pipeline

```
Policy output: [delta_x, delta_y, delta_z, gripper]
    ↓ (range: [-1, 1], scale: 0.02 = 2cm/step)
IK Controller (TODO)
    ↓
Joint position commands
    ↓
SO-101 robot via LeRobot
```

## Key Components

### CameraPreprocessor (`src/deploy/camera.py`)

Handles the visual domain gap between MuJoCo and real camera:
- FOV matching via center crop (130° real → 75° sim)
- Resize to 84×84
- Frame stacking (3 frames)
- CHW format for PyTorch

```python
preprocessor = CameraPreprocessor(
    target_size=(84, 84),
    frame_stack=3,
    sim_fov=75.0,
    real_fov=130.0,
)
```

### PolicyRunner (`src/deploy/policy.py`)

Loads DrQ-v2 checkpoint and runs inference:
- Uses hydra to instantiate agent from config
- Handles frame stacking and batch dimensions
- Returns actions in [-1, 1] range

```python
policy = PolicyRunner(checkpoint_path, device="cuda")
policy.load()
action = policy.get_action(rgb_obs, low_dim_obs)
```

### LowDimStateBuilder (`src/deploy/policy.py`)

Builds the 21-dim state vector expected by the policy:
- joint_pos (6): Joint positions in radians
- joint_vel (6): Joint velocities
- gripper_pos (3): End-effector XYZ position
- gripper_euler (3): End-effector orientation
- cube_pos (3): **NOT available on real robot** - zero-padded

## Gaps for Real Deployment

### 1. IK Controller (Critical)

The sim uses Cartesian control with IK:
```
delta XYZ → IK solver → joint positions
```

Need to port IK from `pick-101/src/controllers/ik_controller.py` or implement for SO-101.

### 2. Forward Kinematics

Currently using hardcoded EE position:
```python
ee_pos = np.array([0.25, 0.0, 0.05])  # Approximate
ee_euler = np.array([0.0, np.pi / 2, 0.0])  # Top-down
```

Need FK to compute actual EE pose from joint angles.

### 3. LeRobot Integration

`RealRobotInterface` is a mock. Need to integrate:
```python
from lerobot.common.robot_devices.robots.so101_follower import SO101Follower
```

### 4. Cube Position

Policy trained with cube_pos in low_dim_state (3 dims). Options:
1. Zero-pad (current approach, may degrade performance)
2. Vision-based cube detection
3. Retrain policy without cube_pos

## Testing

Dry run (no camera/robot required):
```bash
uv run python scripts/rl_inference.py \
    --checkpoint /path/to/best_snapshot.pt \
    --dry_run
```

Output:
```
[1/4] Loading policy...
Loaded checkpoint: /path/to/best_snapshot.pt

[2/4] Initializing camera...
[DRY RUN] Using mock camera observations

[3/4] Connecting to robot...

[4/4] Initializing state builder and action converter...

--- Episode 1/5 ---
  Step 0: action=[-0.99 -0.98  0.97] gripper=1.00
  Step 20: action=[-0.99 -0.99  0.99] gripper=1.00
Episode 1 complete
```

## Dependencies Added

- `opencv-python>=4.8.0` - Camera capture
- `hydra-core>=1.3.0` - Config/agent instantiation
- `natsort>=8.0.0` - Required by robobase

Also requires pick-101 in path for robobase:
```python
sys.path.insert(0, "/home/gota/ggando/ml/pick-101")
sys.path.insert(0, "/home/gota/ggando/ml/pick-101/external/robobase")
```

## Next Steps

1. Port IK controller from pick-101
2. Implement FK for EE state computation
3. Integrate LeRobot for real robot control
4. Test with actual camera
5. Zero-shot transfer test on real robot
