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

## Quick Reference

| Command | Description |
|---------|-------------|
| `lerobot-teleoperate` | Test leader-follower mirroring |
| `lerobot-record` | Record demonstrations (with `--teleop`) |
| `lerobot-train` | Train a policy on recorded data |
| `lerobot-record --policy.path=...` | Run policy inference |

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
