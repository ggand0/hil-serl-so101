# SO-101 Red Cube to Bowl - Training Summary

**Date**: December 13, 2025
**Task**: Pick up the red cube and place it in the green bowl
**Result**: Partial success - policy sometimes completes the task

---

## Training Configuration

### Dataset
- **Repository**: `gtgando/so101_red_cube_to_bowl`
- **Number of Episodes**: 30
- **Total Frames**: 17,591
- **Episode Duration**: ~20 seconds each (with ~5 seconds idle time)
- **FPS**: 30
- **Camera**: gripper_cam (640x480)

### Model Architecture: ACT (Action Chunking with Transformers)
- **Total Parameters**: 51,597,190 (~52M)
- **Vision Backbone**: ResNet-18 (pretrained on ImageNet)
- **Chunk Size**: 100
- **Action Steps**: 100
- **Model Dimensions**: 512
- **Attention Heads**: 8
- **Encoder Layers**: 4
- **Decoder Layers**: 1
- **VAE Latent Dimension**: 32

### Training Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 1e-05
- **Weight Decay**: 0.0001
- **Batch Size**: 8
- **Total Steps**: 100,000
- **Save Frequency**: 20,000 steps
- **Video Backend**: pyav (torchcodec had FFmpeg issues)
- **Device**: CUDA (AMD GPU with ROCm)

---

## Training Results

### Loss Progression
- **Starting Loss**: ~6-7
- **Final Loss**: 0.043
- **Total Improvement**: ~99%
- **Epochs**: 45.48

### Training Duration
- **Start Time**: 16:28 (December 13, 2025)
- **End Time**: 20:02 (December 13, 2025)
- **Total Duration**: ~3.5 hours
- **Update Time**: ~0.124-0.125 seconds/step

### Checkpoints Saved
- Step 20,000
- Step 40,000
- Step 60,000
- Step 80,000
- Step 100,000 (final)

---

## Evaluation Results

### Autonomous Execution Test
- **Observed Behavior**: Policy sometimes successfully picks up the red cube and places it in the green bowl
- **Success Rate**: Partial - works intermittently
- **Failure Modes**: TBD (needs more systematic evaluation)

---

## Commands Used

### Recording Demonstrations
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
    --dataset.repo_id=gtgando/so101_red_cube_to_bowl \
    --dataset.num_episodes=30 \
    --dataset.episode_time_s=20 \
    --dataset.fps=30 \
    --dataset.single_task="Pick up the red cube and place it in the green bowl"
```

### Training
```bash
uv run lerobot-train \
    --dataset.repo_id=gtgando/so101_red_cube_to_bowl \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_red_cube_bowl \
    --job_name=act_so101_red_cube_bowl \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=gtgando/act_so101_red_cube_bowl_policy \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=20000
```

### Evaluation (Policy Inference)
```bash
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=ggando_so101_follower \
    --robot.cameras="{ gripper_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=gtgando/eval_so101_red_cube_bowl \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick up the red cube and place it in the green bowl" \
    --policy.path=gtgando/act_so101_red_cube_bowl_policy
```

---

## Technical Notes

### Issues Encountered

1. **torchcodec FFmpeg error**: Default video backend `torchcodec` failed due to missing FFmpeg libraries. Fixed by using `--dataset.video_backend=pyav`.

2. **Rerun viewer warnings**: Spam of `egui_wgpu` and `re_sdk_comms` warnings. Suppressed with `RUST_LOG=error` or `RUST_LOG=off`.

3. **HF_USER variable**: Must be set before running commands: `export HF_USER=$(huggingface-cli whoami | head -n 1)`

4. **Policy path argument**: Use `--policy.path=` not `--control.policy.path=` for inference.

### Comparison to Previous Training (Green Cube to Paper)

| Metric | Green Cube (Nov 2025) | Red Cube to Bowl (Dec 2025) |
|--------|----------------------|----------------------------|
| Episodes | 26 | 30 |
| Total Frames | 15,858 | 17,591 |
| Final Loss | 0.041 | 0.043 |
| Epochs | 50.45 | 45.48 |
| Training Time | ~3.5 hours | ~3.5 hours |
| Success Rate | Partial | Partial |

---

## Repositories

- **Dataset**: https://huggingface.co/datasets/gtgando/so101_red_cube_to_bowl
- **Model**: https://huggingface.co/gtgando/act_so101_red_cube_bowl_policy
- **Output Directory**: `outputs/train/act_so101_red_cube_bowl/`

---

## Recommendations for Improvement

1. **Reduce episode idle time**: Shorten from 20s to 12-15s to remove ~5s of idle frames
2. **Increase episodes**: Collect 50+ episodes for better generalization
3. **Vary cube positions**: More diverse starting positions
4. **Systematic evaluation**: Run 20+ evaluation episodes to measure actual success rate

---

*Last Updated: December 13, 2025*
