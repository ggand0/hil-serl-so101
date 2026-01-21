# so101-playground

Imitation Learning experiments with the SO-101 robot arm using [LeRobot](https://github.com/huggingface/lerobot).

## Hardware Setup

- **Leader Arm**: SO-101 leader (`/dev/ttyACM1`) - for teleoperation
- **Follower Arm**: SO-101 follower (`/dev/ttyACM0`) - performs the task
- **Camera**: USB webcam mounted on gripper (`/dev/video0`)
- **GPU**: AMD with ROCm 6.4

## Prerequisites

```bash
# Install dependencies
uv sync

# Login to HuggingFace
huggingface-cli login

# Set your HuggingFace username
export HF_USER=$(huggingface-cli whoami | head -n 1)
```

## Imitation Learning Workflow

### 1. Test Teleoperation

First verify the leader-follower setup works:

```bash
uv run lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader
```

### 2. Record Demonstrations

Collect training data by teleoperating the robot:

```bash
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --robot.cameras="{ gripper_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_red_cube_to_bowl \
    --dataset.num_episodes=30 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=10 \
    --dataset.fps=30 \
    --dataset.single_task="Pick up the red cube and place it in the green bowl"
```

**Tips:**
- Press right arrow to end episode early
- Keep episodes short (10-15s) - avoid idle time
- Vary starting positions for better generalization
- Collect 30-50 episodes minimum

### 3. Train the Policy

Train an ACT (Action Chunking with Transformers) policy:

```bash
uv run lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_red_cube_to_bowl \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_red_cube_bowl \
    --job_name=act_so101_red_cube_bowl \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=${HF_USER}/act_so101_red_cube_bowl_policy \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=20000
```

**Expected:**
- Training time: ~3-4 hours
- Initial loss: ~6-7
- Final loss: ~0.04
- Model size: ~207 MB

### 4. Evaluate the Policy

Run the trained policy on the robot:

```bash
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --robot.cameras="{ gripper_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/eval_so101_red_cube_bowl \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Pick up the red cube and place it in the green bowl" \
    --policy.path=${HF_USER}/act_so101_red_cube_bowl_policy
```

Or use a local checkpoint:

```bash
uv run lerobot-record \
    ... \
    --policy.path=outputs/train/act_so101_red_cube_bowl/checkpoints/last/pretrained_model
```

## RL Deployment (Sim-to-Real)

Deploy MuJoCo-trained DrQ-v2 policies to the real robot.

### Calibrate Cube Position

Position the robot to mark where the cube should be placed:

```bash
# Move to training initial position (above cube)
uv run python scripts/ik_reset_position.py

# Also lower to grasp height
uv run python scripts/ik_reset_position.py --lower

# Adjust cube position
uv run python scripts/ik_reset_position.py --cube_x 0.28 --cube_y -0.02

# Read current robot position
uv run python scripts/ik_reset_position.py --record_only
```

### Run RL Inference

Run a trained policy on the real robot:

```bash
# Basic inference
uv run python scripts/rl_inference.py \
    --checkpoint /path/to/snapshot.pt

# With video recording
uv run python scripts/rl_inference.py \
    --checkpoint /path/to/snapshot.pt \
    --record_dir ./recordings \
    --external_camera 2

# Dry run (mock robot/camera)
uv run python scripts/rl_inference.py \
    --checkpoint /path/to/snapshot.pt \
    --dry_run
```

**Key options:**
- `--action_scale`: Meters per action unit (default 0.02 = 2cm)
- `--control_hz`: Control frequency (default 10Hz)
- `--cube_x`, `--cube_y`: Expected cube position

### Test IK Motion

Test the IK controller with simple movements:

```bash
uv run python scripts/test_ik_motion.py
uv run python scripts/test_ik_motion.py --dry_run
```

## HIL-SERL (Human-in-the-Loop RL)

Online RL training with human interventions using DrQ-v2.

### Calibrate Robot

Calibrate leader and follower arms:

```bash
# Calibrate follower
uv run lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower

# Calibrate leader
uv run lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=ggando_so101_leader
```

### Adjust Wrist Angles

Find correct wrist joint angles after recalibration:

```bash
uv run python scripts/adjust_wrist_angles.py
```

Commands: `3 <angle>` (wrist_flex), `4 <angle>` (wrist_roll), `+3/-3`, `+4/-4`, `r` (read), `l` (lift), `q` (quit)

### Record Demonstrations (End-Effector Control)

Record with IK-based end-effector control and locked wrist joints:

```bash
uv run lerobot-record --config outputs/hilserl_drqv2/record_config.json
```

### Start Learner

Start the learner process (runs on GPU, loads offline buffer):

```bash
uv run lerobot-hilserl-learner --config outputs/hilserl_drqv2/train_config.json
```

### Start Actor

Start the actor process (controls robot, sends transitions to learner):

```bash
uv run lerobot-hilserl-actor --config outputs/hilserl_drqv2/train_config.json
```

### Merge Datasets

Merge multiple datasets with episode filtering:

```bash
uv run python scripts/merge_datasets.py
```

Edit the script to configure source datasets and excluded episodes.

## Segmentation Data Collection

Record video footage with random gripper motion for training segmentation models:

```bash
uv run python scripts/record_camera.py --record seg_data.mp4 --random_gripper --duration 60 --preview --camera 2
```

- `--camera`: Camera index (find with `uv run lerobot-find-cameras opencv`)
- `--random_gripper`: Randomly opens/closes gripper, arm torque disabled for manual positioning
- `--preview`: Show live feed while recording
- `--duration`: Recording length in seconds
- `--gripper_interval`: Average time between gripper changes (default: 1.0s)

### IK Grasp Demo

Run automated grasp sequence with video recording:

```bash
uv run python scripts/ik_grasp_demo.py --cube_x 0.25 --cube_y 0.1 --grasp_z 0.03 --output grasp_demo.mp4
```

- `--cube_x`, `--cube_y`: Cube position in meters (Y+ is left)
- `--grasp_z`: Grasp height (default: 0.03m)
- `--camera_index`: Camera index for recording

## Quick Reference

| Command | Description |
|---------|-------------|
| `lerobot-teleoperate` | Test leader-follower mirroring |
| `lerobot-record` | Record demonstrations (with `--teleop`) |
| `lerobot-train` | Train a policy on recorded data |
| `lerobot-record --policy.path=...` | Run IL policy inference |
| `scripts/rl_inference.py` | Run RL policy (sim-to-real) |
| `scripts/ik_reset_position.py` | Calibrate cube position |
| `scripts/test_ik_motion.py` | Test IK controller |
| `scripts/adjust_wrist_angles.py` | Adjust wrist joint angles |
| `scripts/merge_datasets.py` | Merge datasets with filtering |
| `scripts/record_camera.py` | Record camera with random gripper |
| `scripts/depth_estimation.py` | Depth Anything V2 inference |

## Troubleshooting

**Check USB ports:**
```bash
ls /dev/ttyACM*
```

**Find cameras:**
```bash
uv run lerobot-find-cameras opencv
```

**Verify calibration files:**
```bash
ls ~/.cache/huggingface/lerobot/calibration/
```

**Suppress rerun viewer warnings:**
```bash
RUST_LOG=error uv run lerobot-record ...
```

**torchcodec FFmpeg error:** Use `--dataset.video_backend=pyav`

## Experiments

| Task | Dataset | Model | Result |
|------|---------|-------|--------|
| Green cube to paper | [so101_pick_lift_cube](https://huggingface.co/datasets/gtgando/so101_pick_lift_cube) | [act_so101_cube_policy](https://huggingface.co/gtgando/act_so101_cube_policy) | Partial success |
| Red cube to bowl | [so101_red_cube_to_bowl](https://huggingface.co/datasets/gtgando/so101_red_cube_to_bowl) | [act_so101_red_cube_bowl_policy](https://huggingface.co/gtgando/act_so101_red_cube_bowl_policy) | Partial success |

## Documentation

- [RED_CUBE_BOWL_TRAINING.md](RED_CUBE_BOWL_TRAINING.md) - Detailed training guide
- [devlogs/](devlogs/) - Experiment logs and debugging notes

## References

- [LeRobot IL Tutorial](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
