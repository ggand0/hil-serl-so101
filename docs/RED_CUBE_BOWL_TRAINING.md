# Red Cube to Green Bowl - Training Guide

**Task**: Pick up the red cube and place it in the green bowl

## Hardware Setup

- **Leader Arm**: `/dev/ttyACM1` (for teleoperation)
- **Follower Arm**: `/dev/ttyACM0` (performs the task)
- **Camera**: Gripper-mounted camera on follower arm

## Prerequisites

```bash
# Set your HuggingFace username
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER

# Login if needed
huggingface-cli login
```

## Step 1: Record Demonstrations

Record 30-50 demonstration episodes using teleoperation:

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
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the red cube and place it in the green bowl"
```

### Recording Tips
- Collect at least 30-50 episodes (more is better!)
- Vary the cube's starting position
- Maintain consistent grasp and release patterns
- Move slowly and smoothly, especially during placement
- Make sure the cube ends up fully in the bowl

## Step 2: Train the Policy

Train the ACT policy on your dataset:

```bash
uv run lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_red_cube_to_bowl \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_red_cube_bowl \
    --job_name=act_so101_red_cube_bowl \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=${HF_USER}/act_so101_red_cube_bowl_policy
```

### Training Parameters
- **Policy**: ACT (Action Chunking with Transformers)
- **Device**: cuda (AMD ROCm)
- **Checkpoints**: Saved in `outputs/train/act_so101_red_cube_bowl/checkpoints/`

### What to Expect
- Initial loss: ~6-7
- Final loss: ~0.04-0.06 (99%+ improvement)
- Training time: ~3-4 hours
- Model size: ~207 MB

### Resume Training (if interrupted)

```bash
uv run lerobot-train \
    --config_path=outputs/train/act_so101_red_cube_bowl/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

## Step 3: Upload the Trained Model

```bash
huggingface-cli upload ${HF_USER}/act_so101_red_cube_bowl_policy \
    outputs/train/act_so101_red_cube_bowl/checkpoints/last/pretrained_model
```

## Step 4: Evaluate the Policy

Run inference and record evaluation episodes:

```bash
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --robot.cameras="{ gripper_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/eval_so101_red_cube_bowl \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick up the red cube and place it in the green bowl" \
    --policy.path=${HF_USER}/act_so101_red_cube_bowl_policy
```

Or use a local checkpoint:

```bash
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --robot.cameras="{ gripper_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/eval_so101_red_cube_bowl \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick up the red cube and place it in the green bowl" \
    --policy.path=outputs/train/act_so101_red_cube_bowl/checkpoints/last/pretrained_model
```

## Quick Reference

| Step | Command |
|------|---------|
| Teleoperate (test) | `uv run lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=ggando_so101_follower --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=ggando_so101_leader` |
| Record | `uv run lerobot-record ...` |
| Train | `uv run lerobot-train ...` |
| Evaluate | `uv run lerobot-record --policy.path=...` |

## Dataset & Model Repositories

- **Dataset**: https://huggingface.co/datasets/gtgando/so101_red_cube_to_bowl
- **Model**: https://huggingface.co/gtgando/act_so101_red_cube_bowl_policy

## Troubleshooting

- Check USB ports: `ls /dev/ttyACM*`
- Verify calibration files: `ls ~/.cache/huggingface/lerobot/calibration/`
- If camera index 0 doesn't work, try `/dev/video0` or check with `v4l2-ctl --list-devices`
