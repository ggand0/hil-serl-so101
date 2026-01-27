#!/usr/bin/env python3
"""Custom reward classifier training with train/val split, early stopping, and best model selection."""
import argparse
import json
import logging
import random
import shutil
import warnings
from pathlib import Path

# Suppress torchvision video deprecation warning
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_image_transforms(cfg_dict: dict | None) -> ImageTransforms | None:
    """Convert JSON config dict to ImageTransforms object."""
    if cfg_dict is None or not cfg_dict.get("enable", False):
        return None

    # Convert nested tfs dict to ImageTransformConfig objects
    tfs = {}
    for tf_name, tf_cfg in cfg_dict.get("tfs", {}).items():
        tfs[tf_name] = ImageTransformConfig(
            weight=tf_cfg.get("weight", 1.0),
            type=tf_cfg.get("type", "Identity"),
            kwargs=tf_cfg.get("kwargs", {}),
        )

    transforms_cfg = ImageTransformsConfig(
        enable=cfg_dict.get("enable", False),
        max_num_transforms=cfg_dict.get("max_num_transforms", 3),
        random_order=cfg_dict.get("random_order", False),
        tfs=tfs,
    )

    return ImageTransforms(transforms_cfg)


def create_train_val_split(dataset, val_ratio: float = 0.15, seed: int = 42):
    """Random frame-level split for classifier training."""
    num_frames = len(dataset)
    indices = list(range(num_frames))

    random.seed(seed)
    random.shuffle(indices)

    num_val = max(1, int(num_frames * val_ratio))
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    # Log episode distribution
    episode_data_index = dataset.episode_data_index
    num_episodes = len(episode_data_index["from"])

    def get_episode(idx):
        for ep in range(num_episodes):
            if episode_data_index["from"][ep] <= idx < episode_data_index["to"][ep]:
                return ep
        return -1

    val_episodes = set(get_episode(i) for i in val_indices)
    train_episodes = set(get_episode(i) for i in train_indices)

    logger.info(f"Train/Val split: {len(train_indices)} train frames, {len(val_indices)} val frames")
    logger.info(f"Val covers {len(val_episodes)}/{num_episodes} episodes, Train covers {len(train_episodes)}/{num_episodes} episodes")

    return train_indices, val_indices


def evaluate(model, dataloader, device):
    """Evaluate model on dataloader, return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, metrics = model(batch)
            total_loss += loss.item() * batch["next.reward"].size(0)
            correct += metrics["correct"]
            total += metrics["total"]

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0

    model.train()
    return avg_loss, accuracy


def train(
    config_path: str,
    val_ratio: float = 0.15,
    patience: int = 10,
    min_delta: float = 0.001,
):
    """Train reward classifier with validation-based model selection and early stopping."""

    # Load config
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    set_seed(cfg.get("seed", 42))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["policy"].get("device", "cuda"))

    # Create dataset
    logger.info(f"Loading dataset: {cfg['dataset']['repo_id']}")

    # Convert image transforms config dict to ImageTransforms object
    image_transforms = make_image_transforms(cfg["dataset"].get("image_transforms"))

    dataset = LeRobotDataset(
        repo_id=cfg["dataset"]["repo_id"],
        root=cfg["dataset"].get("root"),
        image_transforms=image_transforms,
        video_backend=cfg["dataset"].get("video_backend", "pyav"),
    )

    # Train/val split
    train_indices, val_indices = create_train_val_split(dataset, val_ratio=val_ratio)

    # Save val indices for inspection
    val_indices_file = output_dir / "val_indices.json"
    with open(val_indices_file, "w") as f:
        json.dump({"val_indices": val_indices, "train_indices": train_indices}, f)
    logger.info(f"Saved train/val indices to {val_indices_file}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    logger.info("Creating model...")
    policy_cfg = RewardClassifierConfig(
        num_classes=cfg["policy"].get("num_classes", 2),
        hidden_dim=cfg["policy"].get("hidden_dim", 256),
        latent_dim=cfg["policy"].get("latent_dim", 256),
        image_embedding_pooling_dim=cfg["policy"].get("image_embedding_pooling_dim", 8),
        dropout_rate=cfg["policy"].get("dropout_rate", 0.3),
        model_name=cfg["policy"].get("model_name", "helper2424/resnet10"),
        model_type=cfg["policy"].get("model_type", "cnn"),
        num_cameras=cfg["policy"].get("num_cameras", 1),
        image_size=cfg["policy"].get("image_size", 128),
        learning_rate=cfg["policy"].get("learning_rate", 3e-4),
        weight_decay=cfg["policy"].get("weight_decay", 0.02),
        grad_clip_norm=cfg["policy"].get("grad_clip_norm", 1.0),
        device=str(device),
    )

    # Set input/output features with proper types
    from lerobot.configs.types import FeatureType, PolicyFeature

    input_features = {}
    for key, val in cfg["policy"]["input_features"].items():
        input_features[key] = PolicyFeature(
            type=FeatureType[val["type"]],
            shape=tuple(val["shape"]),
        )
    policy_cfg.input_features = input_features

    output_features = {}
    for key, val in cfg["policy"]["output_features"].items():
        output_features[key] = PolicyFeature(
            type=FeatureType[val["type"]],
            shape=tuple(val["shape"]),
        )
    policy_cfg.output_features = output_features

    model = Classifier(policy_cfg, dataset_stats=dataset.meta.stats)
    model = model.to(device)

    # Optimizer
    lr = cfg["policy"].get("learning_rate", 3e-4)
    weight_decay = cfg["policy"].get("weight_decay", 0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    max_steps = cfg.get("steps", 5000)
    log_freq = cfg.get("log_freq", 50)
    eval_freq = cfg.get("eval_freq", 250)
    save_freq = cfg.get("save_freq", 500)
    grad_clip_norm = cfg["policy"].get("grad_clip_norm", 1.0)

    # Early stopping state
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_step = 0
    best_epoch = 0
    patience_counter = 0

    logger.info(f"Training for max {max_steps} steps with early stopping (patience={patience})")
    logger.info(f"Batch size: {batch_size}, LR: {lr}, Weight decay: {weight_decay}")

    model.train()
    step = 0
    epoch = 0

    while step < max_steps:
        epoch += 1
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            if step >= max_steps:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            loss, metrics = model(batch)
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            step += 1

            # Accumulate epoch stats
            bs = batch["next.reward"].size(0)
            epoch_loss += loss.item() * bs
            epoch_correct += metrics["correct"]
            epoch_total += bs

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{metrics['accuracy']:.1f}%")

            # Periodic checkpoint
            if step % save_freq == 0:
                ckpt_dir = output_dir / "checkpoints" / f"{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir / "pretrained_model")
                tqdm.write(f"Checkpoint saved at step {step}")

        pbar.close()

        # End of epoch - log train metrics
        if epoch_total > 0:
            train_loss = epoch_loss / epoch_total
            train_acc = 100.0 * epoch_correct / epoch_total
            tqdm.write(f"epoch:{epoch} step:{step} train_loss:{train_loss:.4f} train_acc:{train_acc:.1f}%")

        # Validation at end of each epoch
        val_loss, val_acc = evaluate(model, val_loader, device)
        tqdm.write(f"  [VAL] epoch:{epoch} val_loss:{val_loss:.4f} val_acc:{val_acc:.1f}%")

        # Best model selection
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_step = step
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            best_dir = output_dir / "best_model"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            best_dir.mkdir(parents=True)
            model.save_pretrained(best_dir)
            tqdm.write(f"  [BEST] New best model saved at epoch {epoch} (val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%)")
        else:
            patience_counter += 1
            tqdm.write(f"  [PATIENCE] {patience_counter}/{patience} (best was epoch {best_epoch})")

        # Early stopping
        if patience_counter >= patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break

        model.train()

    # Final summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model: epoch {best_epoch} (step {best_step}), val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.1f}%")
    logger.info(f"Best model saved at: {output_dir / 'best_model'}")
    logger.info("=" * 60)

    # Copy best model to final location
    final_dir = output_dir / "final_model"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(output_dir / "best_model", final_dir)
    logger.info(f"Final model copied to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (eval intervals)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Min improvement for best model")
    args = parser.parse_args()

    train(
        config_path=args.config,
        val_ratio=args.val_ratio,
        patience=args.patience,
        min_delta=args.min_delta,
    )
