# SO-101 Camera Setup and Tutorial Preparation

**Date**: November 2, 2025

## Session Overview

Resumed work on SO-101 HIL-SERL project after a week break. Focused on following the official imitation learning tutorial, setting up camera for demonstration recording, and preparing physical setup for cube manipulation tasks.

## Progress Summary

### 1. Reconnection and Context Review

**Issue**: USB port assignments swap on reboot
- `/dev/ttyACM0` and `/dev/ttyACM1` assignments are not persistent across reboots
- This is normal Linux behavior - USB enumeration order is not guaranteed

**Solution Identified**:
- Use stable symlinks: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A017089-if00` and `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A018118-if00`
- Each arm has unique serial number for persistent identification
- For now, manually check ports with `lerobot-find-port` before each session

**Current Port Assignments** (as of this session):
- Follower: `/dev/ttyACM0`
- Leader: `/dev/ttyACM1`

### 2. Following Imitation Learning Tutorial

**Tutorial URL**: https://huggingface.co/docs/lerobot/il_robots

**Key Learning Objectives from Tutorial**:
1. Recording and visualizing datasets
2. Training policies using collected data
3. Evaluating trained policies

**Tutorial Workflow**:
- Teleoperation → Data Recording → Training → Evaluation
- Start with 50+ episodes recommended
- Record 10 variations per object location
- Maintain fixed camera positions
- Ensure manipulated objects are visible in camera feed

### 3. Motor Calibration Issues and Resolution

**Initial Problem**:
- Wrist_roll joint had ~30-degree offset during teleoperation
- Gripper sync was laggy/unresponsive

**Root Cause Analysis**:
Looking at calibration files revealed significant range mismatches:

**Before Recalibration**:
- Leader wrist_roll: range 160-2289 (2129 units)
- Follower wrist_roll: range 0-4095 (4095 units)
- Follower had ~2x the range of leader, causing poor mapping

**Solution**:
- Recalibrated both arms, ensuring middle position during calibration prompt
- Much more careful joint range recording

**After Recalibration** (calibration saved to `./calibration/110225_1`):
- Leader wrist_roll: range 240-4084 (3844 units)
- Follower wrist_roll: range 33-4009 (3976 units)
- Nearly identical ranges now, much better sync

**Calibration Tables Captured**:
```
Leader:
NAME            |    MIN |    POS |    MAX
shoulder_pan    |    752 |   2081 |   3468
shoulder_lift   |    799 |    799 |   3190
elbow_flex      |    937 |   3138 |   3138
wrist_flex      |    807 |   2905 |   3239
wrist_roll      |    240 |   3968 |   4084
gripper         |   1394 |   1465 |   2690

Follower:
NAME            |    MIN |    POS |    MAX
shoulder_pan    |   1030 |   2127 |   3460
shoulder_lift   |    827 |    986 |   3225
elbow_flex      |    925 |   3161 |   3161
wrist_flex      |    954 |   2887 |   3305
wrist_roll      |     53 |   1197 |   4009
gripper         |   1452 |   1487 |   2958
```

**Result**: Wrist_roll sync much more accurate, though not perfect (acceptable for demonstrations)

### 4. Basic Teleoperation Testing

**Command Used**:
```bash
uv run lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader
```

**Results**: ✅ Success
- Follower arm synced smoothly with leader movements
- Wrist joint has minor offset (~30 degrees) but movement tracking works correctly
- All other joints sync well
- Leader arm freely movable (torque disabled)
- Follower mirrors leader (correct direction)

### 5. Camera Setup and Configuration

**Camera Detection**:
```bash
uv run lerobot-find-cameras opencv
```

**Detected Camera**:
- Name: OpenCV Camera @ /dev/video0
- Type: OpenCV
- Backend: V4L2
- Resolution: 640x480 @ 30fps
- USB camera mounted on custom gripper part

**Configuration Added to `env_config_so101.json`**:
```json
"cameras": {
  "gripper_cam": {
    "type": "opencv",
    "index": "/dev/video0",
    "fps": 30,
    "width": 640,
    "height": 480
  }
}
```

Also set `"display_cameras": true` in wrapper config.

### 6. Teleoperation with Camera Visualization

**Command with Camera** (corrected JSON string format):
```bash
uv run lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --robot.cameras="{ gripper_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader \
    --display_data=true
```

**Key Learning**: Camera config must use `index_or_path` parameter (not `index`) when passed as JSON string in CLI.

**Results**: ✅ Success
- Rerun visualization window launched
- Live camera feed visible
- Real-time joint position graphs displayed
- Successfully verified camera positioning and field of view

### 7. Physical Setup Planning

**Task**: Cube pick-and-place demonstration collection

**Available Materials**:
- 2cm colored wooden cubes
- Confirmed follower arm can pick cubes during testing

**Target Zone Options Discussed**:

1. **Colored Paper/Mat** (Recommended for starting) ✅
   - A4 paper is perfect size
   - Use colored cardstock for durability
   - Tape down securely to prevent movement
   - High contrast colors (e.g., red cube → blue target)
   - Setup: 15-20cm distance, 10x10cm target zone
   - Easiest for policy to learn

2. **Small Container/Box** (Next step)
   - Shallow containers 2-5cm tall
   - Square/rectangular better than round
   - Width: 8-12cm (4-6x cube size)
   - Use bright colored plastic
   - Secure with mounting putty

3. **Progressive Difficulty Plan**:
   - Week 1: Colored paper target (simple placement)
   - Week 2: Shallow container (requires precision)
   - Future: Multiple zones, color matching tasks

**Decision**: Start with A4 colored paper approach for first dataset.

### 8. Arm Mounting Solutions

**Problem**: Follower arm tips forward when extended (weight imbalance toward gripper)

**Long-term Solution** (ordered):
- 2x C-clamps (trigger/ratchet style)
- Officially recommended by SO-101 documentation
- Products: IRWIN Quick-Grip, DEWALT Trigger Clamps, or similar
- 4-6 inch size appropriate

**Temporary Solution** (implemented today):
- Heavy weights on both sides of base
- Tape for additional stability
- Non-slip mat underneath
- Keep movements closer to base during testing

### 9. 3D Printed Part Repair

**Issue**: One follower arm 3D printed part broken (split in half)
- Non-critical structural part
- Can continue operation for now

**Glue Recommendations for PLA** (Amazon Japan):
1. **Aron Alpha EXTRA** (アロンアルフア EXTRA) - Most recommended
2. **Loctite Strong Instant Adhesive Pinpointer** (ロックタイト 強力瞬間接着剤)
3. **Cemedine Instant Adhesive** (セメダイン 瞬間接着剤)
4. **Acrisunday** (アクリサンデー) - Acrylic-based

**Repair Tips**:
- Clean surfaces with isopropyl alcohol
- Rough up surfaces with sandpaper
- Apply glue, clamp for 30+ seconds
- Full cure: 24 hours

## Current Status

### ✅ Completed Tasks
1. Basic teleoperation verified working
2. Camera detected and configured
3. Teleoperation with camera visualization successful
4. Calibration refined (wrist_roll sync improved)
5. Physical setup planned and materials identified
6. Temporary arm mounting implemented

### 📋 Todo List Status

- [x] Test basic teleoperation without camera
- [x] Recalibrate to fix wrist_roll offset
- [x] Configure USB camera for follower arm
- [x] Test teleoperation with camera visualization using rerun
- [ ] Set up HuggingFace authentication for dataset upload
- [ ] Record practice dataset (2 episodes for testing)
- [ ] Record full dataset (50+ episodes following tutorial best practices)

### 🔧 Pending Items

1. **HuggingFace Authentication**:
   - Run: `huggingface-cli login`
   - Need write-access token from https://huggingface.co/settings/tokens
   - Required before recording demonstrations

2. **Physical Setup**:
   - Set up colored paper target zone
   - Position camera to see both cube and target
   - Test lighting and visibility

3. **First Recording Test**:
   - Run 2-episode test using `gym_manipulator` command
   - Verify data capture and camera recording
   - Check dataset upload to HuggingFace

## Configuration Files Updated

### `env_config_so101.json`
- Added camera configuration
- Set `display_cameras: true`
- Ready for demonstration recording

### Calibration Files
Current calibration: `~/.cache/huggingface/lerobot/calibration/`
- `robots/so101_follower/ggando_so101_follower.json`
- `teleoperators/so101_leader/ggando_so101_leader.json`

Backup: `./calibration/110225_1/`

## Commands Reference

### Camera Detection
```bash
uv run lerobot-find-cameras opencv
```

### Basic Teleoperation
```bash
uv run lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader
```

### Teleoperation with Camera Visualization
```bash
uv run lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --robot.cameras="{ gripper_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader \
    --display_data=true
```

### Recording Demonstrations
```bash
uv run python -m lerobot.scripts.rl.gym_manipulator --config_path env_config_so101.json
```

## Next Steps

1. **Authenticate with HuggingFace**
   - Generate write-access token
   - Run `huggingface-cli login`

2. **Prepare Physical Setup**
   - Place colored paper target zone
   - Position cube start location
   - Verify camera can see entire workspace

3. **Test Recording** (2 episodes)
   - Practice the pick-and-place motion
   - Verify data capture works
   - Check dataset upload

4. **Full Dataset Collection** (50+ episodes)
   - Record demonstrations following tutorial best practices
   - Maintain consistent setup
   - Introduce variations gradually

## Documentation References

- Main tutorial: https://huggingface.co/docs/lerobot/il_robots
- Camera setup: https://huggingface.co/docs/lerobot/cameras#setup-cameras
- Project overview: `devlogs/101325-so101-hilserl-project-overview.md`
- Debugging history: `devlogs/so101-action-control-debugging.md`

---

*Session completed: November 2, 2025*
*Ready for demonstration recording phase*
