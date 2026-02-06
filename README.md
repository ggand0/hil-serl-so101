# hil-serl-so101

Human-in-the-Loop SERL (HIL-SERL) training for the SO-101 robot arm using [LeRobot](https://github.com/huggingface/lerobot).

Achieves **70% autonomous grasp success rate** after ~3 hours of real-world training (~600 episodes) with human interventions.

**Related repos:**
- [pick-101](https://github.com/ggand0/pick-101) - MuJoCo simulation for sim2real experiments

## Hardware Setup

- **Leader Arm**: SO-101 leader (`/dev/ttyACM1`) - for teleoperation
- **Follower Arm**: SO-101 follower (`/dev/ttyACM0`) - performs the task
- **Camera**: USB webcam mounted on gripper (`/dev/video0`)
- **GPU**: AMD with ROCm 6.4

## Prerequisites

### LeRobot Fork (Required)

This project requires a modified LeRobot with HIL-SERL fixes:

```bash
git clone https://github.com/ggand0/lerobot
cd lerobot
git checkout feat/hil-serl
pip install -e ".[hilserl]"
```

### Project Setup

```bash
# Install dependencies
uv sync

# Login to HuggingFace
huggingface-cli login

# Set your HuggingFace username
export HF_USER=$(huggingface-cli whoami | head -n 1)
```

## HIL-SERL (Human-in-the-Loop RL) ⭐

### Calibrate Robot

Calibrate leader and follower arms:

```bash
# Calibrate follower
uv run lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101_follower

# Calibrate leader
uv run lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_so101_leader
```

### Record Demonstrations

Record demonstrations for the offline buffer:

```bash
uv run python -m lerobot.scripts.rl.gym_manipulator --config_path configs/grasp_only_record_angled_10ep_config.json
```

**Controls:**
- Leader arm controls follower (intervention enabled by default)
- `ESC` - End episode
- `q` - Quit recording

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
- `--model_path` - Path to trained classifier (default: v5_lamp)
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

### Run SAC Policy Inference

Evaluate a trained SAC policy:

```bash
uv run python scripts/hilserl_inference.py \
    --config_path configs/grasp_only_hilserl_eval_config.json \
    --checkpoint outputs/hilserl_grasp_only_v3_lamp/checkpoints/009500/pretrained_model \
    --num_episodes 10
```

With video recording:

```bash
uv run python scripts/hilserl_inference.py \
    --config_path configs/grasp_only_hilserl_eval_config.json \
    --checkpoint outputs/hilserl_grasp_only_v3_lamp/checkpoints/009500/pretrained_model \
    --num_episodes 10 \
    --record_video --video_dir recordings/
```

**Options:**
- `--config_path` - Config JSON (use eval config for fixed reset position)
- `--checkpoint` - Path to `pretrained_model` folder
- `--num_episodes` - Number of episodes to run (default: 10)
- `--device` - Torch device (default: cuda)
- `--record_video` - Record video of each episode
- `--video_dir` - Directory to save videos (default: recordings/)

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
| `python -m lerobot.scripts.rl.learner` | Start SAC learner |
| `python -m lerobot.scripts.rl.actor` | Start SAC actor |
| `scripts/hilserl_inference.py` | Run trained SAC policy |
| `scripts/merge_datasets.py` | Merge datasets with filtering |
| `scripts/utils/unlock_motors.py` | Disable motor torque |

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

**Unlock motors** (if stuck with torque enabled after crash):
```bash
uv run python scripts/utils/unlock_motors.py
uv run python scripts/utils/unlock_motors.py --port /dev/ttyACM1  # Different port
```

## Documentation

- [docs/SIM2REAL.md](docs/SIM2REAL.md) - Sim2real deployment scripts (experimental)
- [docs/CALIBRATION.md](docs/CALIBRATION.md) - Additional calibration scripts
- [docs/DATASETS.md](docs/DATASETS.md) - Dataset creation and labeling
