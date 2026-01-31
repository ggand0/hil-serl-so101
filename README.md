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

### Run Seg+Depth RL Inference

Run DrQ-v2 policy trained with segmentation + depth observations:

```bash
# Full inference with preview
uv run python scripts/rl_inference_seg_depth.py \
    --checkpoint ~/ggando/ml/pick-101/runs/seg_depth_rl/20260119_192807/snapshots/1200000_snapshot.pt \
    --seg_checkpoint ~/ggando/ml/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt \
    --mujoco_mode \
    --cube_x 0.25 --cube_y 0.0 \
    --camera_index 1 \
    --preview \
    --save_preview_video preview.mp4

# With debug output
uv run python scripts/rl_inference_seg_depth.py \
    --checkpoint ~/ggando/ml/pick-101/runs/seg_depth_rl/20260119_192807/snapshots/1200000_snapshot.pt \
    --seg_checkpoint ~/ggando/ml/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt \
    --mujoco_mode \
    --debug_state \
    --cube_x 0.25 --cube_y 0.0

# Dry run (mock robot/camera)
uv run python scripts/rl_inference_seg_depth.py \
    --checkpoint ~/ggando/ml/pick-101/runs/seg_depth_rl/20260119_192807/snapshots/1200000_snapshot.pt \
    --seg_checkpoint ~/ggando/ml/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt \
    --dry_run
```

**Mode flags:**
- `--mujoco_mode`: For policies trained in pick-101 MuJoCo sim (no coordinate transform)
- `--genesis_mode`: For policies trained in Genesis sim (applies coordinate transform)

**Key options:**
- `--seg_checkpoint`: Path to EfficientViT segmentation model
- `--camera_index`: Camera device index (default 0)
- `--preview`: Show live camera/segmentation preview
- `--save_preview_video`: Save preview to video file
- `--debug_state`: Print per-step debug info (seg classes, ee position, actions)
- `--save_obs`: Save observation images for first 5 steps

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

### Record with JSON Config

Record demonstrations using a JSON config file:

```bash
uv run python -m lerobot.scripts.rl.gym_manipulator --config_path configs/grasp_only_record_angled_10ep_config.json
```

**Controls:**
- Leader arm controls follower (intervention enabled by default)
- `ESC` - End episode
- `q` - Quit recording

### Unlock Motors

If motors get stuck with torque enabled (e.g., after a crash):

```bash
uv run python scripts/unlock_motors.py
uv run python scripts/unlock_motors.py --port /dev/ttyACM1  # Different port
```

### Train Reward Classifier

```bash
uv run python scripts/train_reward_classifier.py --config configs/reward_classifier_grasponly_v4_train_config.json
```

Uses custom training script with frame-level train/val split, early stopping, and best model selection.

### Live Reward Classifier Preview

Test the reward classifier with live camera feed:

```bash
uv run python scripts/reward_classifier_live_preview.py
```

**Options:**
- `--model_path` - Path to trained classifier (default: v3)
- `--threshold` - Classification threshold (default: 0.5)
- `--record output.mp4` - Record video

**Controls:**
- `t` - Increase threshold by 0.1
- `r` - Reset threshold to 0.5
- `v` - Toggle recording
- `q` - Quit

### Train with SAC (Reach and Grasp)

Run learner and actor in separate terminals:

**Terminal 1 (Learner):**
```bash
uv run python -m lerobot.scripts.rl.learner --config_path configs/reach_grasp_hilserl_train_config.json
```

**Terminal 2 (Actor):**
```bash
uv run python -m lerobot.scripts.rl.actor --config_path configs/reach_grasp_hilserl_train_config.json
```

### Train with DrQ-v2

Start the learner process (runs on GPU, loads offline buffer):

```bash
uv run lerobot-hilserl-learner --config outputs/hilserl_drqv2/train_config.json
```

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

## Quick Reference

| Command | Description |
|---------|-------------|
| `lerobot-calibrate` | Calibrate robot or teleop arm |
| `lerobot-teleoperate` | Test leader-follower mirroring |
| `lerobot-record` | Record demonstrations (with `--teleop`) |
| `lerobot-train` | Train a policy on recorded data |
| `lerobot-record --policy.path=...` | Run IL policy inference |
| `python -m lerobot.scripts.rl.learner` | Start SAC learner |
| `python -m lerobot.scripts.rl.actor` | Start SAC actor |
| `lerobot-hilserl-learner` | Start DrQ-v2 learner |
| `lerobot-hilserl-actor` | Start DrQ-v2 actor |
| `scripts/rl_inference.py` | Run RGB RL policy (sim-to-real) |
| `scripts/rl_inference_seg_depth.py` | Run seg+depth RL policy (sim-to-real) |
| `scripts/ik_reset_position.py` | Calibrate cube position |
| `scripts/adjust_wrist_angles.py` | Adjust wrist joint angles |
| `scripts/merge_datasets.py` | Merge datasets with filtering |
| `scripts/unlock_motors.py` | Disable motor torque |

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
