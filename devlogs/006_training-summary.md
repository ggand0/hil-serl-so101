# SO-101 ACT Policy Training Summary

**Date**: November 7, 2025
**Task**: Pick up cube and place it on colored paper
**Result**: ✅ Successfully learned manipulation behavior

---

## Training Configuration

### Model Architecture: ACT (Action Chunking with Transformers)

**Policy Type**: Action Chunking with Transformers (ACT)
- **Vision Backbone**: ResNet-18 (pretrained on ImageNet)
- **Model Dimensions**: 512
- **Attention Heads**: 8
- **Feedforward Dimensions**: 3200
- **Encoder Layers**: 4
- **Decoder Layers**: 1
- **Uses VAE**: Yes (Variational Autoencoder for action distribution)
- **VAE Latent Dimension**: 32
- **VAE Encoder Layers**: 4
- **Action Chunk Size**: 100 (predicts 100 future actions at once)
- **Action Steps**: 100
- **Dropout**: 0.1

**Total Parameters**: 51,597,190 (~52M parameters)

### Input Features

**Visual Input**:
- Camera: `gripper_cam` (mounted on follower arm)
- Resolution: 640x480 pixels
- Channels: 3 (RGB)
- Normalization: ImageNet statistics (MEAN_STD)

**State Input**:
- Joint positions: 6 DOF (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- Normalization: MEAN_STD

**Output**:
- Actions: 6 DOF joint commands
- Normalization: MEAN_STD

### Training Hyperparameters

- **Optimizer**: AdamW
- **Learning Rate**: 1e-05 (0.00001)
- **Weight Decay**: 0.0001
- **Gradient Clipping**: 10.0
- **Batch Size**: 8
- **Total Steps**: 100,000
- **KL Weight** (VAE): 10.0
- **Number of Workers**: 4
- **Device**: CUDA (AMD GPU with ROCm 6.4.3)
- **Mixed Precision**: Disabled

### Dataset

- **Repository**: `gtgando/so101_pick_lift_cube`
- **Number of Episodes**: 26
- **Total Frames**: 15,858
- **Task Description**: "Pick up the cube and place it on the colored paper"
- **Video Backend**: PyAV
- **Image Augmentation**: Disabled
- **FPS**: 30 (camera), 10 (robot control)

### Checkpointing

- **Save Frequency**: Every 20,000 steps
- **Checkpoints Saved**:
  - Step 20,000
  - Step 40,000
  - Step 60,000
  - Step 80,000
  - Step 100,000 (final)
- **Output Directory**: `outputs/train/act_so101_cube/`
- **Model Hub**: `gtgando/act_so101_cube_policy`

---

## Training Results

### Loss Progression

| Phase | Step Range | Loss Range | Description |
|-------|-----------|------------|-------------|
| **Early Training** | 0 - 1,200 | 6.827 → 1.790 | Rapid learning phase |
| **Mid Training** | 43,000 - 51,600 | 0.066 → 0.058 | Fine-tuning phase |
| **Late Training** | 97,000 - 100,000 | 0.042 → 0.041 | Convergence phase |

**Starting Loss**: 6.827
**Final Loss**: 0.041
**Total Improvement**: 99.4%

### Training Duration

- **Start Time**: ~08:58 (November 7, 2025)
- **End Time**: ~12:32 (November 7, 2025)
- **Total Duration**: ~3.5 hours
- **Steps per Second**: ~7.94 steps/s
- **Average Update Time**: 0.126 seconds/step
- **Total Epochs**: 50.45 (passed through dataset 50 times)

### Key Milestones

- **Step 200**: Loss dropped to 6.827 (initial rapid descent)
- **Step 1,000**: Loss at 1.972 (74% improvement)
- **Step 20,000**: First checkpoint saved
- **Step 50,000**: Loss at 0.059 (99.1% improvement)
- **Step 100,000**: Final model - Loss at 0.041 (99.4% improvement)

---

## Evaluation Results

### Autonomous Execution Test

**Evaluation Dataset**: `gtgando/eval_so101_pick_lift_cube`
**Number of Episodes Tested**: 5

**Observed Behavior**:
- ✅ Robot successfully locates cube
- ✅ Approaches and attempts to grasp cube
- ✅ Successfully picks up cube after multiple attempts
- ✅ Moves cube toward target (colored paper)
- ⚠️ Gets stuck during placement phase (needs more training data)

**Success Rate**: Partial success - demonstrates learned behavior but needs refinement

**Key Insights**:
- 26 episodes (slightly above minimum) showed good learning
- Pick-up motion well-learned
- Placement motion needs more examples
- Recommend collecting 50+ episodes for production-level performance

---

## Technical Details

### Hardware Setup

**Robot**: SO-101 Follower Arm (TheRobotStudio)
- Motors: 6x STS3215 servos (Feetech)
- Gearing: 1/345 reduction
- DOF: 6 (5 arm joints + 1 gripper)
- Communication: USB Serial (FT232 chipset)

**Camera**: USB Webcam
- Mounted on custom gripper part
- Device: `/dev/video0`
- Backend: V4L2

**Compute**:
- GPU: AMD (ROCm 6.4.3)
- PyTorch: 2.8.0+rocm6.4
- Python: 3.10

### Training Infrastructure

**Framework**: LeRobot v0.3.2 (HuggingFace)
- Based on original ACT implementation
- Integrated with HuggingFace Hub for dataset/model sharing
- Custom SO-101 robot support

**Key Libraries**:
- PyTorch 2.8.0
- torchvision (with PyAV video backend)
- transformers
- datasets

---

## Model Characteristics

### Strengths

1. **Vision-Based Control**: Uses camera feed for spatial awareness
2. **Action Chunking**: Predicts 100 future actions for smooth, coherent motion
3. **Learned Representations**: VAE provides robust action distribution
4. **Pretrained Vision**: ResNet-18 ImageNet weights accelerate learning

### Architecture Highlights

**Transformer Encoder-Decoder**:
- Encoder processes visual features + state
- Decoder generates action sequences
- Multi-head attention (8 heads) captures complex relationships
- Feedforward layers (3200 dims) for expressive transformations

**Variational Autoencoder (VAE)**:
- Encodes action sequences into latent space (32 dims)
- Provides stochastic policy for exploration
- KL divergence regularization prevents mode collapse

---

## Files Generated

### Training Artifacts

```
outputs/train/act_so101_cube/
├── checkpoints/
│   ├── 020000/
│   ├── 040000/
│   ├── 060000/
│   ├── 080000/
│   └── 100000/
│       ├── pretrained_model/
│       │   ├── model.safetensors (207 MB)
│       │   ├── config.json
│       │   └── train_config.json
│       └── training_state/
│           ├── optimizer_param_groups.json
│           └── training_step.json
└── training_loss_curve.png
```

### Datasets

**Training Data**: `gtgando/so101_pick_lift_cube`
- 26 episodes
- Videos: 640x480 @ 30fps
- Total size: ~100 MB

**Evaluation Data**: `gtgando/eval_so101_pick_lift_cube`
- 5 autonomous episodes
- Demonstrates policy behavior
- Total size: ~10 MB

### Model Repository

**HuggingFace Hub**: `gtgando/act_so101_cube_policy`
- Model weights: `model.safetensors` (207 MB)
- Configuration files
- Training metadata
- Public access

---

## Recommendations for Improvement

### Data Collection

1. **Increase Episodes**: Collect 25 more episodes → 50 total
2. **Vary Cube Positions**: 10 episodes per location (5 locations)
3. **Focus on Placement**: Emphasize slow, deliberate placement motion
4. **Consistent Demonstrations**: Maintain similar grasp/release patterns

### Training Adjustments

1. **Longer Training**: Consider 150K-200K steps for 50 episodes
2. **Learning Rate Schedule**: Could add warmup + decay for better convergence
3. **Data Augmentation**: Enable image transforms for robustness

### Evaluation Strategy

1. **Success Metrics**: Define clear success criteria (cube on paper, upright, etc.)
2. **Multiple Runs**: Test 10-20 episodes for statistical significance
3. **Variation Testing**: Try different cube colors, positions, lighting

---

## Conclusion

Successfully trained an ACT policy from 26 demonstration episodes in 3.5 hours. The model achieved 99.4% loss reduction and demonstrates clear understanding of the pick-and-place task. While placement needs refinement, the results validate the approach and show that imitation learning works effectively for the SO-101 robot with limited data.

**Next Steps**:
1. Collect 25 more episodes (→ 50 total)
2. Retrain for 100K steps
3. Evaluate success rate
4. Iterate based on failure modes

---

**Training Loss Curve**: See `training_loss_curve.png`

**Model**: https://huggingface.co/gtgando/act_so101_cube_policy
**Dataset**: https://huggingface.co/datasets/gtgando/so101_pick_lift_cube
**Evaluation**: https://huggingface.co/datasets/gtgando/eval_so101_pick_lift_cube
