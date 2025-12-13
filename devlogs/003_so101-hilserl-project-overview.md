# SO-101 HIL-SERL Project Overview
*Last Updated: January 13, 2025*

## Project Goal
Set up SO-101 robot arms (leader + follower) for HIL-SERL demonstration collection and reinforcement learning, following the [HuggingFace LeRobot HIL-SERL tutorial](https://huggingface.co/docs/lerobot/hilserl).

## Hardware Setup
- **Robot Arms**: SO-101 leader and follower arms (TheRobotStudio)
- **Ports**:
  - Leader: `/dev/ttyACM1`
  - Follower: `/dev/ttyACM0`
- **GPU**: AMD GPU with ROCm 6.4.3
- **Calibration**: Arms calibrated with IDs `ggando_so101_leader` and `ggando_so101_follower`

## What's Working ✅

### 1. Basic Hardware Control
- ✅ Motor calibration completed
- ✅ Motor ID assignment configured
- ✅ Gripper control working (`test_so101_movement.py`)
- ✅ Joint position control verified
- ✅ Working gripper range: ~2.5 (closed) to ~9.0 (open)

### 2. Environment Setup
- ✅ Modern Python environment with `uv`
- ✅ LeRobot installed with `lerobot[hilserl]` extra
- ✅ ROCm PyTorch (2.8.0+rocm6.4) installed for AMD GPU
- ✅ Dependencies: `feetech-servo-sdk`, `numpy`, `pytorch-triton-rocm`

### 3. Script Modifications
Two key LeRobot scripts were modified for SO-101 compatibility:

#### `find_joint_limits_so101_fixed.py`
- Added SO-101 follower and leader imports
- Made URDF/kinematics conditional (SO-101 lacks URDF support)
- Falls back to joint limits only (no end-effector positions)

#### `gym_manipulator_so101_fixed.py`
- Added SO-101 follower import
- Made end-effector step sizes conditional with defaults
- Made kinematics initialization conditional
- Added joint space control fallback
- Made robot reset method conditional
- Implemented leader position communication for direct joint mirroring

### 4. Configuration
Created `env_config_so101.json` with:
- Recording mode for demonstration collection
- SO-101 robot and teleop configurations
- Leader control mode
- AMD GPU device setting (`"device": "cuda"` - works with ROCm)

## Current Issue ⚠️

### StopIteration Error
When attempting to record demonstrations, encountering `StopIteration` error in motor sync_write:

```
File "lerobot/robots/so101_follower/so101_follower.py", line 212, in send_action
    self.bus.sync_write("Goal_Position", goal_pos)
File "lerobot/motors/motors_bus.py", line 1179, in sync_write
    model = next(iter(models))
StopIteration
```

### Root Cause
- Robot expects joint position commands: `{"motor_name.pos": value}`
- Environment sends end-effector commands: `{"delta_x": 0.1, "delta_y": 0.0, "delta_z": -0.05}`
- Missing conversion layer between end-effector actions and joint positions
- SO-101 basic config lacks URDF, so no inverse kinematics available

### Last Fix Attempt
Implemented "leader position communication" approach:
1. GearedLeaderControlWrapper stores leader joint positions when no kinematics available
2. RobotEnv retrieves these positions and sends as direct joint commands
3. Bypasses end-effector control layer entirely

Status: Untested - stopped working on project before verifying fix.

## Key Files

### Scripts
- `test_so101_movement.py` - Basic gripper control test (working)
- `find_joint_limits_so101_fixed.py` - Modified joint limits script
- `gym_manipulator_so101_fixed.py` - Modified demonstration recording script
- `env_config_so101.json` - Configuration for SO-101 setup

### Documentation
- [`so101-gripper-control.md`](./so101-gripper-control.md) - Initial gripper control implementation
- [`so101-hil-serl-compatibility-fixes.md`](./so101-hil-serl-compatibility-fixes.md) - **Detailed code modifications for all SO-101 compatibility fixes**
- [`so101-action-control-debugging.md`](./so101-action-control-debugging.md) - **Deep dive into the StopIteration error and fix attempts**

## Technical Details

### SO-101 vs SO-100 Differences
- SO-101 basic config lacks `urdf_path` and `target_frame_name` attributes
- No end-effector position calculations without URDF
- Requires joint space control instead of end-effector control
- SO-100 `so100_follower_end_effector` variant has URDF support, but no SO-101 equivalent exists

### Modified Code Locations
All modifications in `.venv/lib/python3.10/site-packages/lerobot/`:
- `scripts/rl/gym_manipulator.py` - Lines 60, 342-343, 375-389, 1111-1113, 1158-1163, 1268-1295, 1940-1962
- `scripts/find_joint_limits.py` - Lines 44-45, 52-53, 82-84, 89-96, 104-115, 120-122

### Action Flow
1. **GearedLeaderControlWrapper** reads leader positions
2. **Without kinematics**: Stores positions in `self.unwrapped._leader_positions`
3. **RobotEnv.step()** checks for stored positions
4. **Converts to joint commands**: `{f"{name}.pos": pos}`
5. **Robot.send_action()** sends to motors via `sync_write()`

## Next Steps

### Immediate
1. Test the latest leader position communication fix
2. Move leader arm and verify follower mirrors movements
3. Debug any remaining action format issues

### If Still Broken
1. Consider creating custom joint space control wrapper
2. Investigate if different control mode would work better
3. May need to modify robot's `send_action()` to handle mixed formats

### Once Working
1. Record demonstrations with wooden cubes (ordered but unused)
2. Collect dataset for pick-and-place task
3. Train HIL-SERL policy
4. Deploy and test learned policy

## Hardware Notes
- **Power Supply**: Use separate power sources for each arm to avoid voltage errors
- **Permissions**: USB device permissions needed (dialout group)
- **Calibration Files**: Stored in `~/.cache/huggingface/lerobot/calibration/`

## Dependencies
```toml
dependencies = [
    "feetech-servo-sdk>=1.0.0",
    "lerobot[hilserl]",
    "numpy",
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "torchaudio>=2.4.0",
    "pytorch-triton-rocm>=3.2.0",
]
```

## Command to Run
```bash
uv run python -m lerobot.scripts.rl.gym_manipulator --config_path env_config_so101.json
```

## Known Issues
- ⚠️ `StopIteration` error during demonstration recording (in progress)
- ⚠️ Package reinstall via `uv sync` removes modifications (need to reapply)
- ✅ ROCm PyTorch successfully replaces CUDA version
- ✅ Gymnasium deprecation warnings (cosmetic, not blocking)

---

*Use this document as context when resuming work on this project.*