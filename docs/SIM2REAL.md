# Sim2Real Deployment (Experimental)

Deploy MuJoCo-trained DrQ-v2 policies from [pick-101](https://github.com/ggand0/pick-101) to the real robot.

> **Note:** Direct sim2real transfer did not work due to domain gap. These scripts are kept for potential sim-pretrain → HIL-SERL fine-tuning workflows.

## Calibrate Cube Position

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

## Run RL Inference

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

## Run Seg+Depth RL Inference

Run DrQ-v2 policy trained with segmentation + depth observations:

```bash
# Full inference with preview
uv run python scripts/rl_inference_seg_depth.py \
    --checkpoint /path/to/pick-101/runs/seg_depth_rl/snapshots/snapshot.pt \
    --seg_checkpoint /path/to/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt \
    --mujoco_mode \
    --cube_x 0.25 --cube_y 0.0 \
    --camera_index 1 \
    --preview \
    --save_preview_video preview.mp4

# With debug output
uv run python scripts/rl_inference_seg_depth.py \
    --checkpoint /path/to/pick-101/runs/seg_depth_rl/snapshots/snapshot.pt \
    --seg_checkpoint /path/to/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt \
    --mujoco_mode \
    --debug_state \
    --cube_x 0.25 --cube_y 0.0

# Dry run (mock robot/camera)
uv run python scripts/rl_inference_seg_depth.py \
    --checkpoint /path/to/pick-101/runs/seg_depth_rl/snapshots/snapshot.pt \
    --seg_checkpoint /path/to/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt \
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

## Test IK Motion

Test the IK controller with simple movements:

```bash
uv run python scripts/test_ik_motion.py
uv run python scripts/test_ik_motion.py --dry_run
```
