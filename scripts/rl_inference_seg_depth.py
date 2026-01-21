#!/usr/bin/env python3
"""Real robot inference with seg+depth trained RL policy.

Runs a DrQ-v2 policy trained with segmentation + depth observations
on the real SO-101 robot.

Usage:
    # Dry run (mock robot and camera)
    uv run python scripts/rl_inference_seg_depth.py \
        --checkpoint /path/to/seg_depth_snapshot.pt \
        --seg_checkpoint /path/to/efficientvit_seg/best.ckpt \
        --dry_run

    # Real robot with MuJoCo-trained policy (pick-101)
    uv run python scripts/rl_inference_seg_depth.py \
        --checkpoint ~/ggando/ml/pick-101/runs/seg_depth_rl/.../snapshot.pt \
        --seg_checkpoint ~/ggando/ml/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt \
        --mujoco_mode

    # Real robot with Genesis-trained policy (requires coordinate transform)
    uv run python scripts/rl_inference_seg_depth.py \
        --checkpoint /path/to/genesis_snapshot.pt \
        --seg_checkpoint /path/to/efficientvit_seg/best.ckpt \
        --genesis_mode

Mode selection:
    --mujoco_mode: For policies trained in pick-101 MuJoCo sim (drqv2_lift_seg_depth_*)
                   No coordinate transform, gripper=0.3, wrist locked at pi/2
    --genesis_mode: For policies trained in Genesis sim
                    Applies Genesis->MuJoCo coordinate transform

Architecture:
    Camera → Seg Model → Seg Mask ─┐
            ↓                     ├─→ Stack (2, 84, 84) → Frame Stack (6, 84, 84) ─┐
    Camera → Depth Model → Disparity ──┘                                           │
                                                                                   ├─→ Policy → Cartesian Action → IK → Joint Commands → Robot
    Robot State → FK → low_dim_state ──────────────────────────────────────────────┘
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# Add project root BEFORE any other imports (same as seg_depth_preview.py)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

# Import perception models ONLY at module level (same as seg_depth_preview.py)
from src.deploy.perception import SegmentationModel, DepthModel, MockSegDepthPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Run seg+depth RL policy on real SO-101 robot"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained seg+depth checkpoint (.pt file)",
    )
    parser.add_argument(
        "--seg_checkpoint",
        type=str,
        default="/home/gota/ggando/ml/pick-101/outputs/efficientvit_seg_merged/best-v1.ckpt",
        help="Path to EfficientViT segmentation checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera device index",
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Robot serial port",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=200,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without real robot/camera (mock mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for inference",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=0.02,
        help="Action scale (meters per unit, default 2cm)",
    )
    parser.add_argument(
        "--control_hz",
        type=float,
        default=10.0,
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--cube_x",
        type=float,
        default=0.25,
        help="Expected cube X position (meters)",
    )
    parser.add_argument(
        "--cube_y",
        type=float,
        default=0.0,
        help="Expected cube Y position (meters)",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help="Directory to save episode recordings (enables recording)",
    )
    parser.add_argument(
        "--external_camera",
        type=int,
        default=None,
        help="External camera index for third-person view recording",
    )
    # Simulator-specific flags
    parser.add_argument(
        "--genesis_to_mujoco",
        action="store_true",
        help="Transform actions from Genesis frame to MuJoCo frame",
    )
    parser.add_argument(
        "--genesis_mode",
        action="store_true",
        help="Enable all Genesis-specific fixes: joint offset, coordinate transform, wrist locking",
    )
    parser.add_argument(
        "--mujoco_mode",
        action="store_true",
        help="MuJoCo-trained policy mode: no coord transform, gripper=0.3, wrist locked (for pick-101 policies)",
    )
    parser.add_argument(
        "--pick101_root",
        type=str,
        default="/home/gota/ggando/ml/pick-101",
        help="Path to pick-101 repository (for robobase dependencies)",
    )
    # Debug options
    parser.add_argument(
        "--debug_state",
        action="store_true",
        help="Print verbose state/action debug info at each step",
    )
    parser.add_argument(
        "--save_obs",
        action="store_true",
        help="Save observation images to debug visual mismatch",
    )
    parser.add_argument(
        "--save_obs_video",
        type=str,
        default=None,
        help="Save observation images as video (path to output .mp4)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview GUI of RGB | Seg | Depth during inference",
    )
    parser.add_argument(
        "--save_preview_video",
        type=str,
        default=None,
        help="Save RGB|Seg|Depth preview video (path to output .mp4)",
    )
    parser.add_argument(
        "--debug_seg",
        action="store_true",
        help="Enable debug logging for segmentation model",
    )

    args = parser.parse_args()

    # Add pick-101 paths FIRST (same order as seg_depth_preview.py)
    # CRITICAL: These paths must be added BEFORE any robobase/hydra imports
    pick101_root = Path(args.pick101_root)
    sys.path.insert(0, str(pick101_root))
    sys.path.insert(0, str(pick101_root / "external" / "robobase"))

    # Import deploy modules AFTER pick-101 paths are set up
    # This ensures robobase/hydra imports don't pollute the segmentation inference
    from src.deploy.policy import SegDepthPolicyRunner, LowDimStateBuilder
    from src.deploy.robot import SO101Robot, MockSO101Robot
    from src.deploy.controllers import IKController

    # Derive mode flags
    use_genesis = args.genesis_mode or args.genesis_to_mujoco
    use_mujoco = args.mujoco_mode

    # Coordinate transform only for Genesis-trained policies
    apply_coord_transform = use_genesis and not use_mujoco

    # Wrist locking for both modes (top-down grasp orientation)
    lock_wrist = use_genesis or use_mujoco

    print("=" * 60)
    print("SO-101 RL Inference (DrQ-v2 Seg+Depth)")
    print("=" * 60)
    if use_mujoco:
        print("[MuJoCo mode enabled] (for pick-101 trained policies)")
        print("  - Coordinate transform: None (MuJoCo native)")
        print("  - Locked joints: [3, 4] (both wrist joints)")
        print("  - Gripper reset: 0.3 (partially open)")
    elif use_genesis:
        print("[Genesis mode enabled]")
        print("  - Coordinate transform: Genesis -> MuJoCo")
        print("  - Locked joints: [3, 4] (both wrist joints)")

    # === 1. Initialize Perception (BEFORE policy to avoid import conflicts) ===
    print("\n[1/4] Initializing perception (segmentation + depth)...")
    seg_model = None
    depth_model = None
    cap = None
    mock_preprocessor = None

    if args.dry_run:
        print("  [DRY RUN] Using mock perception")
        mock_preprocessor = MockSegDepthPreprocessor(
            target_size=(84, 84),
            frame_stack=3,
        )
        mock_preprocessor.load_models()
        mock_preprocessor.open_camera()
    else:
        # Load segmentation model FIRST (before policy imports pollute sys.path)
        print("  Loading segmentation model...")
        seg_model = SegmentationModel(args.seg_checkpoint, device=args.device, debug=args.debug_seg)
        if not seg_model.load():
            print("Failed to load segmentation model. Exiting.")
            return

        # Load depth model
        print("  Loading depth model...")
        depth_model = DepthModel(device=args.device)
        if not depth_model.load():
            print("Failed to load depth model. Exiting.")
            return

        # Open camera
        print(f"  Opening camera {args.camera_index}...")
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            print(f"Failed to open camera {args.camera_index}. Exiting.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm up camera and models
        print("  Warming up...")
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                size = min(h, w)
                y_start = (h - size) // 2
                x_start = (w - size) // 2
                frame_cropped = frame[y_start:y_start + size, x_start:x_start + size]
                _ = seg_model.predict(frame_cropped)
                _ = depth_model.predict(frame_cropped)

    # === 2. Load Policy ===
    print("\n[2/4] Loading policy...")
    policy = SegDepthPolicyRunner(args.checkpoint, device=args.device)
    if not policy.load():
        print("Failed to load policy. Exiting.")
        if cap is not None:
            cap.release()
        return
    print(f"  Frame stack: {policy.frame_stack}")
    print(f"  State dim: {policy.state_dim}")

    frame_stack = policy.frame_stack
    frame_buffer = deque(maxlen=frame_stack)

    # Update mock preprocessor frame_stack if needed
    if args.dry_run and mock_preprocessor is not None:
        mock_preprocessor._frame_buffer = deque(maxlen=frame_stack)
        mock_preprocessor.frame_stack = frame_stack

    # === 3. Initialize Robot ===
    print("\n[3/4] Initializing robot...")
    if args.dry_run:
        robot = MockSO101Robot(port=args.robot_port)
        robot.connect()
    else:
        robot = SO101Robot(port=args.robot_port)
        if not robot.connect():
            print("Failed to connect to robot. Exiting.")
            if cap is not None:
                cap.release()
            return

    # === 4. Initialize IK Controller ===
    print("\n[4/4] Initializing IK controller...")
    try:
        ik = IKController()
        print(f"  IK controller ready (damping={ik.damping})")
    except Exception as e:
        print(f"  Failed to initialize IK: {e}")
        if cap is not None:
            cap.release()
        robot.disconnect()
        return

    # State builder (no cube position in real deployment)
    state_builder = LowDimStateBuilder(include_cube_pos=False)

    # === Recording setup ===
    record_dir = None
    external_cap = None
    if args.record_dir:
        record_dir = Path(args.record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Recording] Saving episodes to {record_dir}")

        # Open external camera if specified
        if args.external_camera is not None:
            external_cap = cv2.VideoCapture(args.external_camera)
            if external_cap.isOpened():
                # Warm up
                for _ in range(10):
                    external_cap.read()
                ext_w = int(external_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ext_h = int(external_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  External camera {args.external_camera}: {ext_w}x{ext_h}")
            else:
                print(f"  Failed to open external camera {args.external_camera}")
                external_cap = None

    # Observation video writer (for debugging)
    obs_video_writer = None
    if args.save_obs_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 2-channel visualization: seg (colorized) + depth side by side
        obs_video_writer = cv2.VideoWriter(
            args.save_obs_video, fourcc, args.control_hz, (168, 84)
        )
        print(f"[Debug] Saving observation video to {args.save_obs_video}")

    # Preview video writer (RGB | Seg | Depth)
    preview_video_writer = None
    if args.save_preview_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_size = 240
        # 3 panels side by side: RGB | Seg | Depth
        preview_video_writer = cv2.VideoWriter(
            args.save_preview_video, fourcc, args.control_hz, (preview_size * 3, preview_size)
        )
        print(f"[Recording] Saving preview video to {args.save_preview_video}")

    def center_crop_square(image: np.ndarray) -> np.ndarray:
        """Center crop to square."""
        h, w = image.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return image[y_start : y_start + size, x_start : x_start + size]

    # Frame buffer for stacking low_dim_state
    state_buffer = deque(maxlen=policy.frame_stack)

    # Control timing
    control_dt = 1.0 / args.control_hz

    print("\n" + "=" * 60)
    print("Ready to run. Press Ctrl+C to stop.")
    print("=" * 60)

    # Training initial position (from curriculum_stage=3 in lift_cube.py)
    FINGER_WIDTH_OFFSET = -0.015  # Static finger is offset from gripper center
    GRASP_Z_OFFSET = 0.005
    CUBE_Z = 0.015  # Cube height on table

    # Both MuJoCo and Genesis modes use 30mm above grasp height and partially open gripper
    # (matching curriculum_stage=3 training)
    if use_mujoco or use_genesis:
        HEIGHT_OFFSET = 0.03  # 30mm above grasp height
        RESET_GRIPPER = 0.3  # Partially open (matches gripper_open=0.3 in training)
    else:
        HEIGHT_OFFSET = 0.03  # Default: 3cm above grasp height
        RESET_GRIPPER = 1.0  # Fully open

    # Safe positions
    SAFE_JOINTS = np.zeros(5)  # Extended forward - safe for IK movements
    REST_JOINTS = np.array([-0.0591, -1.8415, 1.7135, 0.7210, -0.1097])  # Folded rest

    def move_to_initial_pose_with_wrist_lock(
        robot, ik, target_pos, num_steps=100, dt=0.05
    ):
        """Move robot to target EE position using IK with wrist locked at pi/2."""
        for step in range(num_steps):
            current_joints = robot.get_joint_positions_radians().copy()

            # Lock wrist joints at pi/2 for top-down
            current_joints[3] = np.pi / 2
            current_joints[4] = np.pi / 2

            # Multiple IK iterations for convergence
            for _ in range(3):
                target_joints = ik.compute_ik(
                    target_pos, current_joints, locked_joints=[3, 4]
                )
                current_joints = target_joints

            # Ensure wrist stays locked
            target_joints[3] = np.pi / 2
            target_joints[4] = np.pi / 2

            robot.send_action(target_joints, RESET_GRIPPER)
            time.sleep(dt)

            # Check convergence
            ik.sync_joint_positions(robot.get_joint_positions_radians())
            ee_pos = ik.get_ee_position()
            error = np.linalg.norm(target_pos - ee_pos)
            if error < 0.005:  # Within 5mm
                break

        return ee_pos

    def safe_return():
        """Safe return sequence: lift up first, then go to rest position."""
        print("\nSafe return sequence...")

        # Step 1: Lift up to safe height
        print("  Lifting to safe height...")
        try:
            ik.sync_joint_positions(robot.get_joint_positions_radians())
            current_ee = ik.get_ee_position()
            safe_height_target = current_ee.copy()
            safe_height_target[2] = 0.15  # Lift to 15cm

            for step in range(40):
                current_joints = robot.get_joint_positions_radians().copy()
                current_joints[3] = np.pi / 2
                current_joints[4] = np.pi / 2
                target_joints = ik.compute_ik(
                    safe_height_target, current_joints, locked_joints=[3, 4]
                )
                target_joints[3] = np.pi / 2
                target_joints[4] = np.pi / 2

                robot.send_action(target_joints, 1.0)
                time.sleep(0.05)

                ik.sync_joint_positions(robot.get_joint_positions_radians())
                ee_pos = ik.get_ee_position()
                if ee_pos[2] > 0.12:
                    break

            print(f"  Lifted to: {ik.get_ee_position()}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Warning: Failed to lift ({e}), going directly to rest...")

        # Step 2: Interpolate to rest position
        print("  Returning to rest position...")
        current_joints = robot.get_joint_positions_radians()
        for i in range(20):
            alpha = (i + 1) / 20
            interp_joints = (1 - alpha) * current_joints + alpha * REST_JOINTS
            robot.send_action(interp_joints, -1.0)
            time.sleep(0.1)
        robot.send_action(REST_JOINTS, -1.0)
        time.sleep(1.0)

    # Segmentation class colors for visualization
    SEG_COLORS = np.array(
        [
            [0, 0, 0],  # 0: unlabeled - black
            [0, 255, 0],  # 1: background - green
            [255, 0, 255],  # 2: table - magenta
            [0, 127, 255],  # 3: cube - orange
            [255, 127, 0],  # 4: static_finger - cyan
            [127, 255, 127],  # 5: moving_finger - light green
        ],
        dtype=np.uint8,
    )

    def visualize_seg_depth_obs(obs: np.ndarray) -> np.ndarray:
        """Create visualization of seg+depth observation.

        Args:
            obs: Observation (2, 84, 84) uint8.

        Returns:
            Visualization (84, 168, 3) BGR image.
        """
        seg_mask = obs[0]  # (84, 84) class IDs 0-5
        disparity = obs[1]  # (84, 84) 0-255

        # Colorize segmentation
        seg_colored = SEG_COLORS[seg_mask]  # (84, 84, 3) RGB

        # Colorize depth (grayscale)
        depth_colored = cv2.applyColorMap(disparity, cv2.COLORMAP_INFERNO)

        # Convert seg to BGR for OpenCV
        seg_bgr = cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR)

        # Side by side
        combined = np.hstack([seg_bgr, depth_colored])
        return combined

    try:
        for episode in range(args.num_episodes):
            print(f"\n--- Episode {episode + 1}/{args.num_episodes} ---")

            # Initialize video writers for this episode
            wrist_writer = None
            external_writer = None
            if record_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                episode_prefix = f"ep{episode+1:02d}_{timestamp}"

                # Wrist camera writer (480x480 square crop)
                if not args.dry_run:
                    wrist_path = record_dir / f"{episode_prefix}_wrist.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    wrist_writer = cv2.VideoWriter(
                        str(wrist_path), fourcc, args.control_hz, (480, 480)
                    )
                    print(f"  Recording wrist: {wrist_path}")

                # External camera writer
                if external_cap is not None:
                    ext_w = int(external_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ext_h = int(external_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ext_size = min(ext_w, ext_h)
                    external_path = record_dir / f"{episode_prefix}_external.mp4"
                    external_writer = cv2.VideoWriter(
                        str(external_path), fourcc, args.control_hz, (ext_size, ext_size)
                    )
                    print(f"  Recording external: {external_path}")

            # Step 1: Reset to safe extended position
            print("  Step 1: Reset position (extended forward)...")
            robot.send_action(SAFE_JOINTS, RESET_GRIPPER)
            time.sleep(3.0)

            # Step 2: Set wrist joints to pi/2 for top-down orientation
            print("  Step 2: Setting top-down wrist orientation...")
            topdown_joints = robot.get_joint_positions_radians().copy()
            topdown_joints[3] = np.pi / 2
            topdown_joints[4] = np.pi / 2
            robot.send_action(topdown_joints, RESET_GRIPPER)
            time.sleep(1.0)

            # Step 3: Move to training initial position with wrist locked
            SAFE_HEIGHT_OFFSET = 0.06  # 60mm above grasp position

            # Step 3a: Move to safe height above target first
            print("  Step 3a: Moving to safe height above target...")
            safe_target = np.array(
                [
                    args.cube_x,
                    args.cube_y + FINGER_WIDTH_OFFSET,
                    CUBE_Z + GRASP_Z_OFFSET + SAFE_HEIGHT_OFFSET,
                ]
            )
            print(f"    Safe target: {safe_target}")

            ee_pos = move_to_initial_pose_with_wrist_lock(robot, ik, safe_target)
            print(f"    Reached: {ee_pos}")

            # Step 3b: Lower to final position (for both MuJoCo and Genesis modes)
            if use_mujoco or use_genesis:
                print("  Step 3b: Lowering to grasp height...")
                final_target = np.array(
                    [
                        args.cube_x,
                        args.cube_y + FINGER_WIDTH_OFFSET,
                        CUBE_Z + GRASP_Z_OFFSET + HEIGHT_OFFSET,
                    ]
                )
                print(f"    Final target: {final_target}")
                ee_pos = move_to_initial_pose_with_wrist_lock(robot, ik, final_target)
                print(f"    Reached: {ee_pos}")

            # Reset buffers
            state_buffer.clear()
            frame_buffer.clear()

            # Fill perception buffer with initial frames
            if args.dry_run:
                mock_preprocessor.reset()
                if not mock_preprocessor.fill_buffer():
                    print("  Failed to fill perception buffer, skipping episode")
                    continue
            else:
                # Flush stale camera frames
                for _ in range(5):
                    cap.read()
                time.sleep(0.1)

                # Fill buffer with initial observations
                for i in range(frame_stack):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"  Failed to capture frame {i+1}/{frame_stack}")
                        break
                    h, w = frame.shape[:2]
                    size = min(h, w)
                    y_start = (h - size) // 2
                    x_start = (w - size) // 2
                    frame_cropped = frame[y_start:y_start + size, x_start:x_start + size]
                    seg_mask = seg_model.predict(frame_cropped)
                    disparity = depth_model.predict(frame_cropped)
                    seg_mask_resized = cv2.resize(seg_mask, (84, 84), interpolation=cv2.INTER_NEAREST)
                    disparity_resized = cv2.resize(disparity, (84, 84), interpolation=cv2.INTER_LINEAR)
                    # Remap real segmentation classes to sim classes (shift by 1)
                    seg_mask_remapped = np.clip(seg_mask_resized.astype(np.int32) - 1, 0, 4).astype(np.uint8)
                    obs_frame = np.stack([seg_mask_remapped, disparity_resized], axis=0).astype(np.uint8)
                    frame_buffer.append(obs_frame)

                if len(frame_buffer) < frame_stack:
                    print("  Failed to fill perception buffer, skipping episode")
                    continue

            # Get initial robot state and compute FK
            joint_pos_rad = robot.get_joint_positions_radians()
            joint_vel = robot.get_joint_velocities()
            gripper_state = robot.get_gripper_position()

            # Use IK controller for FK
            ik.sync_joint_positions(joint_pos_rad)
            ee_pos = ik.get_ee_position()
            ee_euler = ik.get_ee_euler()

            print(f"  Initial EE position: {ee_pos}")
            print(f"  Initial joints (rad): {joint_pos_rad}")

            # Fill state buffer
            for _ in range(policy.frame_stack):
                state = state_builder.build(
                    joint_pos=joint_pos_rad,
                    joint_vel=joint_vel,
                    gripper_pos=ee_pos,
                    gripper_euler=ee_euler,
                    gripper_state=gripper_state,
                )
                state_buffer.append(state)

            # Episode loop
            for step in range(args.episode_length):
                step_start = time.time()

                # Get seg+depth observation - DIRECT CAPTURE (like seg_depth_preview.py)
                if args.dry_run:
                    seg_depth_obs = mock_preprocessor.get_stacked_observation()
                    last_raw_frame = None
                    last_seg_mask = None
                    last_disparity = None
                else:
                    # Capture frame directly
                    ret, frame = cap.read()
                    if not ret:
                        print("  Camera read failed, ending episode")
                        break
                    last_raw_frame = frame.copy()

                    # Debug: raw frame stats
                    if args.debug_seg and step < 3:
                        print(f"\n  [INFERENCE DEBUG] Step {step}")
                        print(f"    Raw frame: shape={frame.shape}, dtype={frame.dtype}")
                        print(f"    Raw frame range: [{frame.min()}, {frame.max()}]")
                        print(f"    Raw frame mean BGR: [{frame[:,:,0].mean():.1f}, {frame[:,:,1].mean():.1f}, {frame[:,:,2].mean():.1f}]")

                    # Center crop to square (exactly like seg_depth_preview.py)
                    h, w = frame.shape[:2]
                    size = min(h, w)
                    y_start = (h - size) // 2
                    x_start = (w - size) // 2
                    frame_cropped = frame[y_start:y_start + size, x_start:x_start + size]

                    # Debug: cropped frame stats
                    if args.debug_seg and step < 3:
                        print(f"    Cropped: shape={frame_cropped.shape}, y_start={y_start}, x_start={x_start}, size={size}")
                        print(f"    Cropped mean BGR: [{frame_cropped[:,:,0].mean():.1f}, {frame_cropped[:,:,1].mean():.1f}, {frame_cropped[:,:,2].mean():.1f}]")

                    # Run inference directly (exactly like seg_depth_preview.py)
                    seg_mask = seg_model.predict(frame_cropped)
                    disparity = depth_model.predict(frame_cropped)
                    last_seg_mask = seg_mask
                    last_disparity = disparity

                    # Debug: inference output stats
                    if args.debug_seg and step < 3:
                        unique, counts = np.unique(seg_mask, return_counts=True)
                        class_dist = dict(zip(unique.tolist(), counts.tolist()))
                        print(f"    Seg mask: shape={seg_mask.shape}, dtype={seg_mask.dtype}")
                        print(f"    Seg class distribution: {class_dist}")
                        print(f"    Disparity: shape={disparity.shape}, dtype={disparity.dtype}, range=[{disparity.min()}, {disparity.max()}]")

                    # Resize to 84x84
                    seg_mask_resized = cv2.resize(seg_mask, (84, 84), interpolation=cv2.INTER_NEAREST)
                    disparity_resized = cv2.resize(disparity, (84, 84), interpolation=cv2.INTER_LINEAR)

                    # Remap real segmentation classes to sim classes
                    # Real: 0=unlabeled, 1=bg, 2=table, 3=cube, 4=static, 5=moving
                    # Sim:  0=bg, 1=table, 2=cube, 3=static, 4=moving
                    seg_mask_remapped = np.clip(seg_mask_resized.astype(np.int32) - 1, 0, 4).astype(np.uint8)

                    # Debug: resized output stats
                    if args.debug_seg and step < 3:
                        unique_r, counts_r = np.unique(seg_mask_remapped, return_counts=True)
                        class_dist_r = dict(zip(unique_r.tolist(), counts_r.tolist()))
                        print(f"    Seg remapped: shape={seg_mask_remapped.shape}, classes={class_dist_r}")
                        print(f"    Disp resized: shape={disparity_resized.shape}, range=[{disparity_resized.min()}, {disparity_resized.max()}]")

                    # Stack as (2, 84, 84)
                    obs_frame = np.stack([seg_mask_remapped, disparity_resized], axis=0).astype(np.uint8)

                    # Update frame buffer
                    frame_buffer.append(obs_frame)

                    # Get stacked observation
                    if len(frame_buffer) < frame_stack:
                        # Fill buffer if not full
                        while len(frame_buffer) < frame_stack:
                            frame_buffer.append(obs_frame)
                    seg_depth_obs = np.stack(list(frame_buffer), axis=0)

                if seg_depth_obs is None:
                    print("  Perception frame error, ending episode")
                    break

                # Save observation for debugging
                if args.save_obs and step < 5:
                    obs_viz = visualize_seg_depth_obs(seg_depth_obs[-1])
                    cv2.imwrite(f"seg_depth_obs_ep{episode+1}_step{step}.png", obs_viz)
                    print(
                        f"    Saved observation: seg_depth_obs_ep{episode+1}_step{step}.png"
                    )

                # Write to observation video
                if obs_video_writer is not None:
                    obs_viz = visualize_seg_depth_obs(seg_depth_obs[-1])
                    obs_video_writer.write(obs_viz)

                # Record wrist camera (get raw frame for higher quality)
                if wrist_writer is not None and last_raw_frame is not None:
                    cropped = center_crop_square(last_raw_frame)
                    resized = cv2.resize(cropped, (480, 480))
                    wrist_writer.write(resized)

                # Record external camera
                if external_writer is not None and external_cap is not None:
                    ret, ext_frame = external_cap.read()
                    if ret:
                        cropped = center_crop_square(ext_frame)
                        external_writer.write(cropped)

                # Show preview GUI and/or record preview video
                if (args.preview or preview_video_writer is not None) and not args.dry_run:
                    # Build preview image directly
                    preview_size = 240
                    seg_colors = np.array([
                        [0, 0, 0],        # 0: unlabeled - black
                        [128, 128, 128],  # 1: background - gray
                        [0, 128, 0],      # 2: table - green
                        [0, 165, 255],    # 3: cube - orange
                        [255, 0, 0],      # 4: static_finger - blue
                        [255, 0, 255],    # 5: moving_finger - magenta
                    ], dtype=np.uint8)

                    # RGB panel
                    rgb_cropped = center_crop_square(last_raw_frame)
                    rgb_panel = cv2.resize(rgb_cropped, (preview_size, preview_size))

                    # Seg panel
                    if last_seg_mask is not None:
                        seg_colored = seg_colors[last_seg_mask]
                        seg_panel = cv2.resize(seg_colored, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST)
                    else:
                        seg_panel = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)

                    # Depth panel
                    if last_disparity is not None:
                        depth_colored = cv2.applyColorMap(last_disparity, cv2.COLORMAP_INFERNO)
                        depth_panel = cv2.resize(depth_colored, (preview_size, preview_size))
                    else:
                        depth_panel = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)

                    preview = np.hstack([rgb_panel, seg_panel, depth_panel])

                    # Add step info overlay
                    cv2.putText(
                        preview,
                        f"Ep {episode+1} Step {step}",
                        (10, preview.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    # Write to video
                    if preview_video_writer is not None:
                        preview_video_writer.write(preview)
                    # Show GUI
                    if args.preview:
                        cv2.imshow("Seg+Depth Preview (q to quit)", preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("  Preview quit requested")
                            raise KeyboardInterrupt

                # Stack low_dim states
                low_dim_obs = np.stack(list(state_buffer), axis=0).astype(np.float32)

                # Debug state output - extended to track drift behavior
                if args.debug_state and step < 20:
                    # Compact per-step summary
                    seg_ch = seg_depth_obs[-1, 0]  # Latest frame seg
                    unique, counts = np.unique(seg_ch, return_counts=True)
                    seg_summary = {k: v for k, v in zip(unique.tolist(), counts.tolist())}
                    latest_state = low_dim_obs[-1]
                    ee_xyz = latest_state[12:15]
                    print(f"\n  [DEBUG] Step {step}:")
                    print(f"    seg: {seg_summary}")
                    print(f"    ee_xyz: [{ee_xyz[0]:.4f}, {ee_xyz[1]:.4f}, {ee_xyz[2]:.4f}]")

                    if step == 0:
                        print(f"    joint_pos (6): {latest_state[0:6]}")
                        print(f"    joint_vel (6): {latest_state[6:12]}")
                        print(f"    gripper_euler (3): {latest_state[15:18]}")

                # Slice low_dim_obs to match policy's expected state_dim
                if low_dim_obs.shape[-1] != policy.state_dim:
                    low_dim_obs = low_dim_obs[..., : policy.state_dim]

                # Get action from policy
                action = policy.get_action(seg_depth_obs, low_dim_obs)

                # Debug action output
                if args.debug_state and step < 20:
                    print(f"    action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] grip={action[3]:.3f}")

                # Parse action
                delta_xyz = action[:3].copy()
                gripper_action = action[3]

                # Transform from Genesis frame to MuJoCo frame (only for Genesis-trained policies)
                if apply_coord_transform:
                    genesis_x, genesis_y, genesis_z = delta_xyz
                    delta_xyz = np.array(
                        [
                            -genesis_y,  # MuJoCo X (forward) = -Genesis Y
                            genesis_x,  # MuJoCo Y (sideways) = Genesis X
                            genesis_z,  # Z unchanged
                        ]
                    )
                    if args.debug_state and step < 10:
                        print(
                            f"    [DEBUG] Transformed: genesis({genesis_x:.3f},{genesis_y:.3f},{genesis_z:.3f}) -> mujoco({delta_xyz[0]:.3f},{delta_xyz[1]:.3f},{delta_xyz[2]:.3f})"
                        )

                # Use IK to convert Cartesian action to joint targets
                current_joints = robot.get_joint_positions_radians().copy()

                target_joints = ik.cartesian_to_joints(
                    delta_xyz,
                    current_joints,
                    action_scale=args.action_scale,
                    locked_joints=[3, 4] if lock_wrist else [4],
                )

                # Debug IK
                if args.debug_state and step < 5:
                    ik.sync_joint_positions(current_joints)
                    current_ee = ik.get_ee_position()
                    target_ee = current_ee + delta_xyz * args.action_scale
                    joint_diff = target_joints - current_joints
                    print(
                        f"    [IK DEBUG] current_ee=[{current_ee[0]:.4f},{current_ee[1]:.4f},{current_ee[2]:.4f}]"
                    )
                    print(
                        f"    [IK DEBUG] target_ee=[{target_ee[0]:.4f},{target_ee[1]:.4f},{target_ee[2]:.4f}]"
                    )
                    print(f"    [IK DEBUG] joint_diff={joint_diff}")

                # Send action to robot
                if not args.dry_run:
                    robot.send_action(target_joints, gripper_action)

                # Update state for next step
                joint_pos_rad = robot.get_joint_positions_radians()
                joint_vel = robot.get_joint_velocities()
                gripper_state = robot.get_gripper_position()

                # FK for EE state
                ik.sync_joint_positions(joint_pos_rad)
                ee_pos = ik.get_ee_position()
                ee_euler = ik.get_ee_euler()

                state = state_builder.build(
                    joint_pos=joint_pos_rad,
                    joint_vel=joint_vel,
                    gripper_pos=ee_pos,
                    gripper_euler=ee_euler,
                    gripper_state=gripper_state,
                )
                state_buffer.append(state)

                # Status
                if step % 20 == 0:
                    if use_mujoco or use_genesis:
                        print(
                            f"  Step {step}: delta=[{delta_xyz[0]:.2f},{delta_xyz[1]:.2f},{delta_xyz[2]:.2f}] "
                            f"gripper={gripper_action:.2f} ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]"
                        )
                    else:
                        print(
                            f"  Step {step}: delta={delta_xyz} "
                            f"gripper={gripper_action:.2f} ee_z={ee_pos[2]:.3f}"
                        )

                # Control rate
                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Close video writers for this episode
            if wrist_writer is not None:
                wrist_writer.release()
            if external_writer is not None:
                external_writer.release()

            print(f"  Episode {episode + 1} complete")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        # Close any open writers
        if "wrist_writer" in dir() and wrist_writer is not None:
            wrist_writer.release()
        if "external_writer" in dir() and external_writer is not None:
            external_writer.release()

    finally:
        print("\nCleaning up...")
        if args.preview:
            cv2.destroyAllWindows()
        if obs_video_writer is not None:
            obs_video_writer.release()
            print(f"  Saved observation video: {args.save_obs_video}")
        if preview_video_writer is not None:
            preview_video_writer.release()
            print(f"  Saved preview video: {args.save_preview_video}")
        if cap is not None:
            cap.release()
        if external_cap is not None:
            external_cap.release()

        # Wait for serial port to recover from interrupted communication
        time.sleep(0.5)

        try:
            safe_return()
        except Exception as e:
            print(f"  Warning: safe_return failed ({e})")

        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
