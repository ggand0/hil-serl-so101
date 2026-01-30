#!/usr/bin/env python3
"""Live preview with reward classifier inference overlay and optional teleoperation."""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

from lerobot.policies.sac.reward_model.modeling_classifier import Classifier


def setup_teleoperation(
    follower_port: str,
    leader_port: str,
    calibration_dir: str,
    follower_id: str = "ggando_so101_follower",
    leader_id: str = "ggando_so101_leader",
):
    """Initialize leader and follower robots for teleoperation."""
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    calibration_path = Path(calibration_dir)

    # Initialize follower
    follower_config = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        calibration_dir=calibration_path / "robots" / "so101_follower",
        use_degrees=True,
    )
    follower = make_robot_from_config(follower_config)
    follower.connect()

    # Initialize leader
    leader_config = SO101LeaderConfig(
        port=leader_port,
        id=leader_id,
        calibration_dir=calibration_path / "teleoperators" / "so101_leader",
        use_degrees=True,
    )
    leader = make_teleoperator_from_config(leader_config)
    leader.connect()

    return follower, leader


def main():
    parser = argparse.ArgumentParser(description="Live reward classifier preview")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/gota/ggando/ml/so101-playground/outputs/reward_classifier_grasp_only_v4/best_model",
        help="Path to trained reward classifier",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="/dev/video0",
        help="Camera device path or index",
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
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Output video file path (e.g., output.mp4)",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Enable teleoperation (leader controls follower)",
    )
    parser.add_argument(
        "--follower_port",
        type=str,
        default="/dev/ttyACM0",
        help="Follower robot port",
    )
    parser.add_argument(
        "--leader_port",
        type=str,
        default="/dev/ttyACM1",
        help="Leader robot port",
    )
    parser.add_argument(
        "--calibration_dir",
        type=str,
        default="/home/gota/.cache/huggingface/lerobot/calibration",
        help="Calibration directory",
    )
    parser.add_argument(
        "--follower_id",
        type=str,
        default="ggando_so101_follower",
        help="Follower robot ID (must match calibration file)",
    )
    parser.add_argument(
        "--leader_id",
        type=str,
        default="ggando_so101_leader",
        help="Leader robot ID (must match calibration file)",
    )
    args = parser.parse_args()

    # Load model directly using Classifier.from_pretrained (same as eval script)
    print(f"Loading model from {args.model_path}...")
    model = Classifier.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
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

    # Setup teleoperation if enabled
    follower = None
    leader = None
    if args.teleop:
        print("Setting up teleoperation...")
        try:
            follower, leader = setup_teleoperation(
                args.follower_port,
                args.leader_port,
                args.calibration_dir,
                args.follower_id,
                args.leader_id,
            )
            print("Teleoperation ready - leader controls follower")
        except Exception as e:
            print(f"Failed to setup teleoperation: {e}")
            print("Continuing without teleoperation...")
            args.teleop = False

    print("\nControls:")
    print("  'q' - Quit")
    print("  't' - Increase threshold (+0.1)")
    print("  'r' - Reset threshold to 0.5")
    print("  'v' - Toggle recording")
    if args.teleop:
        print("  Leader arm controls follower")

    threshold = args.threshold

    # Video recording setup
    video_writer = None
    recording = False
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.record, fourcc, 30.0, (640, 480))
        recording = True
        print(f"Recording to {args.record}")

    while True:
        # Teleoperation step
        if args.teleop and follower is not None and leader is not None:
            try:
                action = leader.get_action()
                follower.send_action(action)
            except Exception as e:
                print(f"Teleop error: {e}")

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Preprocess frame for classifier
        # Center crop 640x480 -> 480x480
        h, w = frame.shape[:2]
        crop_x = (w - 480) // 2
        cropped = frame[:, crop_x:crop_x+480]

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb_frame, (args.image_size, args.image_size))

        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Create batch dict for normalization
        batch = {
            "observation.images.gripper_cam": img_tensor,
            "next.reward": torch.zeros(1, 1).to(device),  # Dummy for normalization
        }

        # Run inference
        with torch.no_grad():
            # Normalize and get images
            batch_norm = model.normalize_inputs(batch)
            images = [batch_norm["observation.images.gripper_cam"]]

            # Get predictions
            outputs = model.predict(images)
            prob = outputs.probabilities.item()

        # Draw results on frame
        is_success = prob > threshold

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

        # Recording indicator
        if recording:
            cv2.circle(frame, (620, 20), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (580, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Teleop indicator
        if args.teleop:
            cv2.putText(frame, "TELEOP", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Write frame to video
        if recording and video_writer is not None:
            video_writer.write(frame)

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
        elif key == ord("v"):
            if video_writer is None:
                # Start recording
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                record_path = args.record or f"recording_{int(cv2.getTickCount())}.mp4"
                video_writer = cv2.VideoWriter(record_path, fourcc, 30.0, (640, 480))
                recording = True
                print(f"Started recording to {record_path}")
            else:
                recording = not recording
                print(f"Recording: {'ON' if recording else 'PAUSED'}")

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print("Video saved")
    cv2.destroyAllWindows()

    # Cleanup teleoperation
    if follower is not None:
        try:
            follower.disconnect()
            print("Follower disconnected")
        except Exception:
            pass
    if leader is not None:
        try:
            leader.disconnect()
            print("Leader disconnected")
        except Exception:
            pass


if __name__ == "__main__":
    main()
