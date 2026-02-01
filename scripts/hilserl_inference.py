#!/usr/bin/env python3
"""HIL-SERL inference script for running trained SAC policies on real robot.

Usage:
    # Run inference with checkpoint
    uv run python scripts/hilserl_inference.py \
        --config_path configs/grasp_only_hilserl_train_config.json \
        --checkpoint outputs/hilserl_grasp_only_v2/checkpoints/002500/pretrained_model

    # Specify number of episodes
    uv run python scripts/hilserl_inference.py \
        --config_path configs/grasp_only_hilserl_train_config.json \
        --checkpoint outputs/hilserl_grasp_only_v2/checkpoints/last/pretrained_model \
        --num_episodes 5
"""

import argparse
import logging
import sys
import time

import numpy as np
import torch

# Add lerobot to path
sys.path.insert(0, "/home/gota/ggando/ml/lerobot/src")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="HIL-SERL inference on real robot")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (pretrained_model folder)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device",
    )
    args = parser.parse_args()

    # Import after path setup
    from lerobot.policies.sac.modeling_sac import SACPolicy
    from lerobot.scripts.rl.gym_manipulator import make_robot_env
    from lerobot.configs.train import TrainRLServerPipelineConfig
    import draccus

    print("=" * 60)
    print("HIL-SERL Inference")
    print("=" * 60)
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}")

    # === 1. Load Config ===
    print("\n[1/3] Loading config...")
    # Use draccus.load() to avoid CLI arg conflicts
    with open(args.config_path) as f:
        cfg = draccus.load(TrainRLServerPipelineConfig, f)
    print(f"  Task: {cfg.env.task}")
    print(f"  FPS: {cfg.env.fps}")

    # === 2. Load Policy ===
    print("\n[2/3] Loading SAC policy...")
    policy = SACPolicy.from_pretrained(args.checkpoint)
    policy.to(args.device)
    policy.eval()
    print(f"  Policy loaded on {args.device}")

    # === 3. Create Environment ===
    print("\n[3/3] Creating environment...")
    env = make_robot_env(cfg.env)
    print(f"  Environment created")
    print(f"  Action space: {env.action_space}")

    # === Run inference ===
    print("\n" + "=" * 60)
    print("Starting inference. Press Ctrl+C to stop.")
    print("Use teleop device to intervene if needed.")
    print("=" * 60)

    fps = cfg.env.fps
    control_dt = 1.0 / fps

    try:
        for episode in range(args.num_episodes):
            print(f"\n--- Episode {episode + 1}/{args.num_episodes} ---")

            obs, info = env.reset()
            done = False
            truncated = False
            step = 0
            episode_reward = 0.0

            while not done and not truncated:
                step_start = time.perf_counter()

                # Debug: print obs keys on first step
                if step == 0:
                    print(f"  Observation keys: {list(obs.keys())}")
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                        elif isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, dtype={v.dtype} (tensor)")

                # Prepare observation for policy
                # Env already returns tensors with batch dim [1, ...], just move to device
                obs_batch = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        tensor = torch.from_numpy(value).float()
                        if tensor.dim() == 3:  # Image without batch: [C, H, W]
                            tensor = tensor.unsqueeze(0)
                        elif tensor.dim() == 1:  # State without batch: [N]
                            tensor = tensor.unsqueeze(0)
                        obs_batch[key] = tensor.to(args.device)
                    elif isinstance(value, torch.Tensor):
                        # Already has batch dim from env, just ensure float and device
                        obs_batch[key] = value.float().to(args.device)

                # Get action from policy
                with torch.no_grad():
                    action = policy.select_action(batch=obs_batch)

                # Ensure action is a tensor (env expects torch tensor)
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action).float()

                # Step environment
                next_obs, reward, done, truncated, info = env.step(action)

                episode_reward += reward
                step += 1

                # Check for intervention
                is_intervention = info.get("is_intervention", False)
                if is_intervention:
                    action = info.get("action_intervention", action)

                # Status logging - convert to numpy for display
                if step % 10 == 0:
                    action_np = action.cpu().numpy().flatten() if isinstance(action, torch.Tensor) else action
                    gripper_action = action_np[-1] if len(action_np) > 3 else 0
                    intervention_str = " [TELEOP]" if is_intervention else ""
                    print(
                        f"  Step {step}: reward={reward:.3f} "
                        f"action=[{action_np[0]:.2f},{action_np[1]:.2f},{action_np[2]:.2f}] "
                        f"gripper={gripper_action:.2f}{intervention_str}"
                    )

                obs = next_obs

                # Control rate
                elapsed = time.perf_counter() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            status = "SUCCESS" if reward > 0.5 else "FAIL"
            print(f"  Episode {episode + 1} complete: steps={step}, reward={episode_reward:.2f} [{status}]")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("\nCleaning up...")
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
