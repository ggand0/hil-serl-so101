# Calibration & Setup

Additional calibration scripts for specific setups.

## Adjust Wrist Angles

Find correct wrist joint angles after recalibration (useful for locked wrist IK setups):

```bash
uv run python scripts/adjust_wrist_angles.py
```

**Commands:**
- `3 <angle>` - Set wrist_flex angle
- `4 <angle>` - Set wrist_roll angle
- `+3/-3` - Increment/decrement wrist_flex
- `+4/-4` - Increment/decrement wrist_roll
- `r` - Read current position
- `l` - Lift arm
- `q` - Quit

## Record with End-Effector Control (Legacy)

For IK-based end-effector control with locked wrist joints:

```bash
uv run lerobot-record --config outputs/hilserl_drqv2/record_config.json
```

This mode uses inverse kinematics to convert end-effector deltas to joint commands while keeping wrist joints fixed.
