#!/usr/bin/env python3
"""Live preview with reward classifier inference overlay."""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def main():
    parser = argparse.ArgumentParser(description="Live reward classifier preview")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/gota/ggando/ml/so101-playground/outputs/reward_classifier_reach_grasp/checkpoints/002000/pretrained_model",
        help="Path to trained reward classifier",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="/dev/video0",
        help="Camera device path or index",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/gota/.cache/huggingface/lerobot/gtgando/so101_reach_grasp_cube_reward",
        help="Path to dataset for normalization stats",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Input image size for classifier",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.pretrained_path = args.model_path

    # Load dataset metadata for normalization stats
    ds_meta = LeRobotDatasetMetadata(
        repo_id="gtgando/so101_reach_grasp_cube_reward",
        root=Path(args.dataset_path),
    )

    # Create policy
    policy = make_policy(cfg=config, ds_meta=ds_meta)
    policy.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    print(f"Model loaded on {device}")

    # Open camera
    if args.camera.isdigit():
        cap = cv2.VideoCapture(int(args.camera))
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit")
    print("Press 't' to adjust threshold (+0.1)")
    print("Press 'r' to reset threshold to 0.5")

    threshold = args.threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Preprocess frame for classifier
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb_frame, (args.image_size, args.image_size))

        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Create batch dict
        batch = {
            "observation.images.gripper_cam": img_tensor,
            "next.reward": torch.zeros(1, 1).to(device),  # Dummy for normalization
        }

        # Run inference
        with torch.no_grad():
            pred = policy.predict_reward(batch, threshold=threshold)

            # Get probability
            batch_normalized = policy.normalize_inputs(batch)
            images = [batch_normalized["observation.images.gripper_cam"]]
            output = policy.predict(images)
            prob = output.probabilities.item()

        # Draw results on frame
        is_success = pred.item() > 0.5

        # Draw probability bar
        bar_width = 200
        bar_height = 30
        bar_x = 10
        bar_y = 10

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Probability fill
        fill_width = int(bar_width * prob)
        color = (0, 255, 0) if is_success else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Threshold line
        thresh_x = bar_x + int(bar_width * threshold)
        cv2.line(frame, (thresh_x, bar_y - 5), (thresh_x, bar_y + bar_height + 5), (255, 255, 0), 2)

        # Text
        label = "SUCCESS" if is_success else "FAILURE"
        cv2.putText(
            frame,
            f"{label}: {prob:.2%}",
            (bar_x, bar_y + bar_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Threshold: {threshold:.2f}",
            (bar_x, bar_y + bar_height + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show preprocessed view (small)
        preview_size = 128
        preview = cv2.resize(rgb_frame, (preview_size, preview_size))
        preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

        # Place preview in corner
        frame[frame.shape[0] - preview_size - 10:frame.shape[0] - 10,
              frame.shape[1] - preview_size - 10:frame.shape[1] - 10] = preview
        cv2.rectangle(
            frame,
            (frame.shape[1] - preview_size - 10, frame.shape[0] - preview_size - 10),
            (frame.shape[1] - 10, frame.shape[0] - 10),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Model Input",
            (frame.shape[1] - preview_size - 5, frame.shape[0] - preview_size - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        # Display frame
        cv2.imshow("Reward Classifier Live Preview", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            threshold = min(1.0, threshold + 0.1)
            print(f"Threshold: {threshold:.2f}")
        elif key == ord("r"):
            threshold = 0.5
            print(f"Threshold reset to {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
