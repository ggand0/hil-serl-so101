#!/usr/bin/env python3
"""Evaluate reward classifier on validation set."""
import argparse
import json
import warnings

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*")

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config JSON")
    parser.add_argument("--model_path", default=None, help="Model path (default: best_model in output_dir)")
    parser.add_argument("--val_json", default=None, help="Val episodes JSON (default: val_episodes.json in output_dir)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = json.load(f)

    output_dir = cfg["output_dir"]
    model_path = args.model_path or f"{output_dir}/best_model"
    val_json = args.val_json or f"{output_dir}/val_episodes.json"

    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id=cfg["dataset"]["repo_id"],
        root=cfg["dataset"].get("root"),
        video_backend=cfg["dataset"].get("video_backend", "pyav"),
    )

    # Load val indices
    with open(val_json) as f:
        val_episodes = json.load(f)

    val_indices = []
    for ep_str, frame_indices in val_episodes.items():
        ep_idx = int(ep_str)
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        for frame_idx in frame_indices:
            val_indices.append(ep_start + frame_idx)

    print(f"Val set: {len(val_indices)} frames from {len(val_episodes)} episodes")

    # Load model
    print(f"Loading model from {model_path}")
    model = Classifier.from_pretrained(model_path)
    device = cfg["policy"].get("device", "cuda")
    model = model.to(device)
    model.eval()

    # Create val loader
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Run inference
    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    all_probs = []
    all_labels = []

    print(f"\nRunning inference (threshold={args.threshold})...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Get labels before normalization
            labels = batch["next.reward"].squeeze()

            # Normalize and get images
            batch_norm = model.normalize_inputs(batch)
            images = [batch_norm[key] for key in model.config.input_features if key.startswith("observation.images")]

            # Get predictions
            outputs = model.predict(images)
            probs = outputs.probabilities
            preds = (probs > args.threshold).float()

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Confusion matrix
            labels_int = labels.long()
            preds_int = preds.long()
            tp += ((preds_int == 1) & (labels_int == 1)).sum().item()
            fp += ((preds_int == 1) & (labels_int == 0)).sum().item()
            tn += ((preds_int == 0) & (labels_int == 0)).sum().item()
            fn += ((preds_int == 0) & (labels_int == 1)).sum().item()

    # Compute metrics
    acc = 100.0 * correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"Val Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:    0      1")
    print(f"  Actual 0:   {tn:4d}   {fp:4d}")
    print(f"  Actual 1:   {fn:4d}   {tp:4d}")
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"{'=' * 50}")

    # Class distribution
    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    print(f"\nClass distribution: {n_neg} negative, {n_pos} positive ({100*n_pos/len(all_labels):.1f}% positive)")


if __name__ == "__main__":
    main()
