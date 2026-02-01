#!/usr/bin/env python
"""
Behavioral Cloning (BC) pretraining for SAC policy.

This script pretrains the SAC actor network using supervised learning on demonstration data
before starting reinforcement learning. This gives the policy a better starting point.

Usage:
    python scripts/bc_pretrain_sac.py --config configs/reach_grasp_hilserl_train_config.json --bc_steps 5000

The pretrained model will be saved to the output_dir specified in config, under 'bc_pretrained/'.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add lerobot to path
sys.path.insert(0, "/home/gota/ggando/ml/lerobot/src")

# Import robot/camera/teleop modules to register choice classes
from lerobot.cameras import opencv  # noqa: F401
from lerobot.robots import so100_follower, so101_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> TrainRLServerPipelineConfig:
    """Load training config from JSON file."""
    cfg = TrainRLServerPipelineConfig.from_pretrained(config_path)
    return cfg


def create_bc_dataloader(
    dataset: LeRobotDataset,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader from demonstration dataset."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


def preprocess_batch(batch: dict, policy: SACPolicy, device: torch.device,
                     resize_size: tuple = (128, 128), crop_params: dict = None) -> tuple[dict, torch.Tensor]:
    """Preprocess batch for BC training."""
    # Get observation keys from policy config
    obs_keys = list(policy.config.input_features.keys())

    # Build observation dict
    observations = {}
    for key in obs_keys:
        if key in batch:
            obs = batch[key].to(device)
            # Handle image observations
            if "image" in key:
                # Dataset stores images in [0, 1] - keep as is
                # The policy's encoder handles normalization internally

                # Apply cropping if specified
                if crop_params and key in crop_params:
                    y, x, h, w = crop_params[key]
                    obs = obs[..., y:y+h, x:x+w]

                # Resize to match policy input size
                if resize_size:
                    obs = F.interpolate(obs, size=resize_size, mode='bilinear', align_corners=False)

            observations[key] = obs

    # Get actions
    actions = batch["action"].to(device)

    return observations, actions


def compute_bc_loss(
    policy: SACPolicy,
    observations: dict,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Compute behavioral cloning loss (MSE between predicted and demo actions).

    Note: The actor's encoder normalizes observations internally, so we pass raw observations.
    """
    # Get observation features if using frozen encoder (normalize=True for external call)
    observation_features = None
    if policy.config.vision_encoder_name is not None and policy.config.freeze_vision_encoder:
        with torch.no_grad():
            observation_features = policy.actor.encoder.get_cached_image_features(
                observations, normalize=True
            )

    # Forward through actor to get predicted actions (means)
    # Actor encoder normalizes observations internally
    _, _, means = policy.actor(observations, observation_features)

    # Normalize target actions to match the output space of the actor
    action_key = list(policy.config.output_features.keys())[0]
    normalized_actions = policy.normalize_targets({action_key: actions})[action_key]

    # MSE loss
    bc_loss = F.mse_loss(means, normalized_actions)

    return bc_loss


def bc_pretrain(
    config_path: str,
    bc_steps: int = 5000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    save_freq: int = 1000,
    num_workers: int = 4,
):
    """Run BC pretraining."""
    setup_logging()

    # Load config
    logging.info(f"Loading config from {config_path}")
    cfg = load_config(config_path)

    device = torch.device(cfg.policy.device if hasattr(cfg.policy, "device") else "cuda")
    logging.info(f"Using device: {device}")

    # Create output directory for BC pretrained model
    bc_output_dir = os.path.join(cfg.output_dir, "bc_pretrained")
    os.makedirs(bc_output_dir, exist_ok=True)
    logging.info(f"BC pretrained model will be saved to: {bc_output_dir}")

    # Load dataset first to get metadata
    logging.info(f"Loading dataset: {cfg.dataset.repo_id}")
    if cfg.dataset.root and os.path.exists(cfg.dataset.root):
        logging.info(f"Loading from local path: {cfg.dataset.root}")
        dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            video_backend=cfg.dataset.video_backend,
        )
    else:
        logging.info("Downloading dataset from hub")
        dataset = make_dataset(cfg)
    logging.info(f"Dataset size: {len(dataset)} samples")

    # Convert dataset stats from numpy to tensor (required by NormalizeBuffer)
    def convert_stats_to_tensor(stats):
        converted = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                converted[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        converted[key][k] = torch.from_numpy(v).float()
                    elif isinstance(v, torch.Tensor):
                        converted[key][k] = v.float()
                    else:
                        converted[key][k] = v
            else:
                converted[key] = value
        return converted

    # Create a modified metadata with tensor stats
    class MetaWithTensorStats:
        def __init__(self, meta):
            self.features = meta.features
            self.stats = convert_stats_to_tensor(meta.stats)

    tensor_meta = MetaWithTensorStats(dataset.meta)

    # Create policy with dataset metadata
    logging.info("Creating SAC policy")
    policy = make_policy(cfg.policy, ds_meta=tensor_meta)
    policy = policy.to(device)
    policy.train()

    # Only train the actor
    for param in policy.parameters():
        param.requires_grad = False
    for param in policy.actor.parameters():
        param.requires_grad = True

    # If encoder is not frozen, also train it
    if not policy.config.freeze_vision_encoder:
        for param in policy.encoder_actor.parameters():
            param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Create optimizer for actor only
    actor_params = list(policy.actor.parameters())
    if not policy.config.freeze_vision_encoder:
        actor_params += list(policy.encoder_actor.parameters())

    optimizer = torch.optim.Adam(actor_params, lr=learning_rate)

    # Create dataloader from already loaded dataset
    dataloader = create_bc_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Get image preprocessing params from config
    crop_params = None
    resize_size = (128, 128)  # Default
    if hasattr(cfg, 'env') and hasattr(cfg.env, 'wrapper'):
        crop_params = getattr(cfg.env.wrapper, 'crop_params_dict', None)
        rs = getattr(cfg.env.wrapper, 'resize_size', None)
        if rs:
            resize_size = tuple(rs)
    logging.info(f"Image preprocessing: crop={crop_params}, resize={resize_size}")

    # Training loop
    logging.info(f"Starting BC pretraining for {bc_steps} steps")
    step = 0
    epoch = 0
    running_loss = 0.0

    pbar = tqdm(total=bc_steps, desc="BC Pretraining")

    while step < bc_steps:
        epoch += 1
        for batch in dataloader:
            if step >= bc_steps:
                break

            # Preprocess batch
            observations, actions = preprocess_batch(batch, policy, device, resize_size, crop_params)

            # Compute BC loss
            optimizer.zero_grad()
            loss = compute_bc_loss(policy, observations, actions)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=40.0)

            optimizer.step()

            # Update stats
            running_loss += loss.item()
            step += 1
            pbar.update(1)

            # Logging
            if step % 100 == 0:
                avg_loss = running_loss / 100
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "epoch": epoch})
                running_loss = 0.0

            # Save checkpoint
            if step % save_freq == 0:
                checkpoint_dir = os.path.join(bc_output_dir, f"step_{step:06d}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Save policy state dict
                torch.save(policy.state_dict(), os.path.join(checkpoint_dir, "policy.pt"))

                # Save as pretrained model format
                policy.save_pretrained(os.path.join(checkpoint_dir, "pretrained_model"))

                logging.info(f"Saved checkpoint at step {step}")

    pbar.close()

    # Save final model
    final_dir = os.path.join(bc_output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(final_dir, "policy.pt"))
    policy.save_pretrained(os.path.join(final_dir, "pretrained_model"))
    logging.info(f"Saved final BC pretrained model to {final_dir}")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("BC Pretraining Complete!")
    print("=" * 60)
    print(f"\nTo use this pretrained model for HIL-SERL training:")
    print(f"1. Add to your config:")
    print(f'   "pretrained_path": "{final_dir}/pretrained_model"')
    print(f"\nOr copy the pretrained_model to your checkpoint directory.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BC pretrain SAC policy")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON file",
    )
    parser.add_argument(
        "--bc_steps",
        type=int,
        default=5000,
        help="Number of BC training steps (default: 5000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for BC training (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )

    args = parser.parse_args()

    bc_pretrain(
        config_path=args.config,
        bc_steps=args.bc_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
