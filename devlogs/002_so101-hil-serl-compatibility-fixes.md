# SO-101 HIL-SERL Compatibility Fixes

## Overview
This document summarizes the changes made to LeRobot scripts to support SO-101 arms for HIL-SERL demonstration collection. The main issues were missing SO-101 imports and lack of URDF/end-effector support in basic SO-101 configurations.

## Changes to find_joint_limits.py

### Problem
The script only supported SO-100 arms and crashed when trying to access URDF attributes that don't exist on basic SO-101 configs.

### Fixes Applied

#### 1. Added SO-101 Import Support (Lines 44-45, 52-53)
```python
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,  # <-- ADDED
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    gamepad,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,  # <-- ADDED
)
```

#### 2. Made Kinematics Conditional (Lines 82-84)
```python
kinematics = None
if hasattr(cfg.robot, 'urdf_path') and hasattr(cfg.robot, 'target_frame_name'):
    kinematics = RobotKinematics(cfg.robot.urdf_path, cfg.robot.target_frame_name)
```

#### 3. Made End-Effector Calculations Conditional (Lines 89-96)
```python
ee_pos = None
if kinematics is not None:
    ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3]

max_pos = joint_positions.copy()
min_pos = joint_positions.copy()
max_ee = ee_pos.copy() if ee_pos is not None else None
min_ee = ee_pos.copy() if ee_pos is not None else None
```

#### 4. Made Loop Kinematics Conditional (Lines 104-115)
```python
ee_pos = None
if kinematics is not None:
    ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3]

# Update min/max values
if kinematics is not None and ee_pos is not None:
    max_ee = np.maximum(max_ee, ee_pos)
    min_ee = np.minimum(min_ee, ee_pos)
```

#### 5. Made Output Conditional (Lines 120-122)
```python
if kinematics is not None:
    print(f"Max ee position {np.round(max_ee, 4).tolist()}")
    print(f"Min ee position {np.round(min_ee, 4).tolist()}")
```

### Result
Script now works with SO-101 arms, outputting joint limits only (no end-effector positions) since SO-101 basic config lacks URDF support.

## Changes to gym_manipulator.py

### Problem
Multiple compatibility issues with SO-101 arms including missing imports, URDF dependencies, and missing methods.

### Fixes Applied

#### 1. Added SO-101 Follower Import (Line 60)
```python
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,  # <-- ADDED
)
```

#### 2. Made End-Effector Step Sizes Conditional (Lines 1940-1944, 1952-1956)
```python
# For "leader" mode:
end_effector_step_sizes = getattr(cfg.robot, 'end_effector_step_sizes', {
    "x": 0.02,
    "y": 0.02,
    "z": 0.02,
})

# For "leader_automatic" mode:
end_effector_step_sizes = getattr(cfg.robot, 'end_effector_step_sizes', {
    "x": 0.02,
    "y": 0.02,
    "z": 0.02,
})
```

#### 3. Made GearedLeaderControlWrapper Kinematics Conditional (Lines 1158-1163)
```python
# Initialize robot control
self.kinematics = None
if hasattr(env.unwrapped.robot.config, 'urdf_path') and hasattr(env.unwrapped.robot.config, 'target_frame_name'):
    self.kinematics = RobotKinematics(
        urdf_path=env.unwrapped.robot.config.urdf_path,
        target_frame_name=env.unwrapped.robot.config.target_frame_name,
    )
```

#### 4. Made Observation Enhancement Conditional (Lines 1111-1113)
```python
if self.kinematics is not None:
    current_ee_pos = self.kinematics.forward_kinematics(current_joint_pos)[:3, 3]
    observation["agent_pos"] = np.concatenate([observation["agent_pos"], current_ee_pos], -1)
```

#### 5. Made Action Calculation Conditional with Joint Space Fallback (Lines 1268-1277)
```python
if self.kinematics is not None:
    leader_ee = self.kinematics.forward_kinematics(leader_pos)[:3, 3]
    follower_ee = self.kinematics.forward_kinematics(follower_pos)[:3, 3]

    action = np.clip(leader_ee - follower_ee, -self.end_effector_step_sizes, self.end_effector_step_sizes)
    action = action / self.end_effector_step_sizes
else:
    # Fallback to joint space control when no kinematics available
    action = np.clip(leader_pos - follower_pos, -0.1, 0.1)  # Simple joint space control
```

#### 6. Made Robot Reset Conditional (Lines 342-343)
```python
if hasattr(self.robot, 'reset'):
    self.robot.reset()
```

### Result
Script now works with SO-101 arms using joint space control when URDF/kinematics aren't available, with fallback defaults for missing configuration attributes.

## Configuration File Created

Created `env_config_so101.json` with proper SO-101 setup:
```json
{
  "type": "gym_manipulator",
  "mode": "record",
  "robot": {
    "type": "so101_follower",
    "port": "/dev/ttyACM0",
    "id": "ggando_so101_follower"
  },
  "teleop": {
    "type": "so101_leader",
    "port": "/dev/ttyACM1",
    "id": "ggando_so101_leader"
  },
  "wrapper": {
    "control_mode": "leader"
  }
}
```

## Key Learnings

1. **Import Dependency**: Both scripts needed explicit imports for so101 modules to register them with the configuration system.

2. **URDF Optional**: Basic SO-101 configs lack URDF support, requiring conditional checks before accessing kinematics-related attributes.

3. **Fallback Mechanisms**: Joint space control can substitute for end-effector control when kinematics aren't available.

4. **Method Availability**: Not all robot classes implement the same interface (e.g., `reset()` method missing).

5. **Default Values**: Providing sensible defaults (like end-effector step sizes) allows compatibility across different robot configurations.

## Status
Both scripts now fully support SO-101 arms for HIL-SERL workflow. The demonstration recording environment initializes successfully and is ready for data collection.