#!/usr/bin/env python3
"""Real robot inference with trained RL policy.

Runs a trained DrQ-v2 policy on the real SO-101 robot.

Usage:
    # Dry run (mock robot and camera)
    uv run python scripts/rl_inference.py --checkpoint /path/to/snapshot.pt --dry_run

    # Real robot
    uv run python scripts/rl_inference.py --checkpoint /path/to/snapshot.pt

    # With video recording (wrist + external camera)
    uv run python scripts/rl_inference.py --checkpoint /path/to/snapshot.pt \
        --record_dir ./recordings --external_camera 2

Architecture:
    Camera → Preprocess → Frame Stack ─┐
                                       ├─→ Policy → Cartesian Action → IK → Joint Commands → Robot
    Robot State → FK → low_dim_state ──┘
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import local deploy modules first (before adding pick-101 which also has 'src')
from src.deploy.camera import CameraPreprocessor
from src.deploy.policy import PolicyRunner, LowDimStateBuilder
from src.deploy.robot import SO101Robot, MockSO101Robot
from src.deploy.controllers import IKController

# Note: pick-101 paths are added after arg parsing to use --pick101_root


def main():
    parser = argparse.ArgumentParser(description="Run RL policy on real SO-101 robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt file)",
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
    # Genesis-specific flags
    parser.add_argument(
        "--genesis_to_mujoco",
        action="store_true",
        help="Transform actions from Genesis frame to MuJoCo frame. "
             "Genesis: X=sideways, -Y=forward, Z=up. "
             "MuJoCo: X=forward, Y=sideways, Z=up. "
             "Transformation: MuJoCo_X=-Genesis_Y, MuJoCo_Y=Genesis_X",
    )
    parser.add_argument(
        "--genesis_mode",
        action="store_true",
        help="Enable all Genesis-specific fixes: joint offset, coordinate transform, "
             "wrist locking, and grasp height reset. Equivalent to setting multiple flags.",
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
        "--rotate_image",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate camera image by degrees (for orientation mismatch)",
    )
    # Sim2real diagnostic options
    parser.add_argument(
        "--sim_frames_dir",
        type=str,
        default=None,
        help="Directory with pre-recorded sim frames (PNG/NPY) for sim2real diagnostic",
    )
    parser.add_argument(
        "--sim_states_file",
        type=str,
        default=None,
        help="NPY file with sim low_dim_states for sim2real diagnostic",
    )

    args = parser.parse_args()

    # Add pick-101 paths for robobase and training modules
    pick101_root = Path(args.pick101_root)
    sys.path.insert(0, str(pick101_root))
    sys.path.insert(0, str(pick101_root / "external" / "robobase"))

    # Derive use_genesis flag for convenience
    use_genesis = args.genesis_mode or args.genesis_to_mujoco

    print("=" * 60)
    print("SO-101 RL Inference (DrQ-v2)")
    print("=" * 60)
    if use_genesis:
        print("[Genesis mode enabled]")
        print("  - Coordinate transform: Genesis -> MuJoCo")
        print("  - Joint offset correction: elbow_flex -12.5 deg")
        print("  - Locked joints: [3, 4] (both wrist joints)")

    # === 1. Load Policy ===
    print("\n[1/4] Loading policy...")
    policy = PolicyRunner(args.checkpoint, device=args.device)
    if not policy.load():
        print("Failed to load policy. Exiting.")
        return
    print(f"  Frame stack: {policy.frame_stack}")

    # === 2. Initialize Camera ===
    print("\n[2/4] Initializing camera...")
    camera = None
    use_mock_camera = args.dry_run

    if not use_mock_camera:
        camera = CameraPreprocessor(
            target_size=(84, 84),
            frame_stack=policy.frame_stack,
            camera_index=args.camera_index,
        )
        if not camera.open():
            print("  No camera available, using mock observations.")
            use_mock_camera = True
            camera = None
        else:
            camera.warm_up()
            print(f"  Camera opened at index {args.camera_index}")
    else:
        print("  [DRY RUN] Using mock camera observations")

    # === 3. Initialize Robot ===
    print("\n[3/4] Initializing robot...")
    if args.dry_run:
        robot = MockSO101Robot(port=args.robot_port)
        robot.connect()
    else:
        robot = SO101Robot(port=args.robot_port)
        if not robot.connect():
            print("Failed to connect to robot. Exiting.")
            if camera is not None:
                camera.close()
            return

    # === 4. Initialize IK Controller ===
    print("\n[4/4] Initializing IK controller...")
    try:
        ik = IKController()
        print(f"  IK controller ready (damping={ik.damping})")
    except Exception as e:
        print(f"  Failed to initialize IK: {e}")
        if camera is not None:
            camera.close()
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

    # === Load sim frames if provided (for sim2real diagnostic) ===
    sim_frames = None
    sim_states = None
    if args.sim_frames_dir:
        sim_frames_dir = Path(args.sim_frames_dir)
        if not sim_frames_dir.exists():
            print(f"ERROR: Sim frames directory not found: {sim_frames_dir}")
            if camera is not None:
                camera.close()
            robot.disconnect()
            return

        print(f"\n[SIM2REAL TEST] Loading frames from {sim_frames_dir}")

        # Check for batch frames.npy first (preferred)
        frames_npy = sim_frames_dir / "frames.npy"
        if frames_npy.exists():
            sim_frames = np.load(frames_npy)
            print(f"  Loaded frames.npy: {sim_frames.shape}")
        else:
            # Fall back to individual PNGs
            frame_files = sorted(sim_frames_dir.glob("*.png"))
            if len(frame_files) == 0:
                print(f"ERROR: No frames.npy or PNG files found in {sim_frames_dir}")
                if camera is not None:
                    camera.close()
                robot.disconnect()
                return

            print(f"  Loading {len(frame_files)} PNG files...")
            sim_frames = []
            for f in frame_files:
                img_bgr = cv2.imread(str(f))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (84, 84):
                    img_rgb = cv2.resize(img_rgb, (84, 84), interpolation=cv2.INTER_AREA)
                frame = np.transpose(img_rgb, (2, 0, 1)).astype(np.uint8)
                sim_frames.append(frame)
            sim_frames = np.array(sim_frames)
            print(f"  Loaded frames shape: {sim_frames.shape}")

    if args.sim_states_file:
        sim_states_path = Path(args.sim_states_file)
        if not sim_states_path.exists():
            print(f"ERROR: Sim states file not found: {sim_states_path}")
            if camera is not None:
                camera.close()
            robot.disconnect()
            return
        sim_states = np.load(sim_states_path)
        print(f"  Loaded sim states shape: {sim_states.shape}")

    use_sim_obs = sim_frames is not None

    # Observation video writer (for debugging)
    obs_video_writer = None
    if args.save_obs_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        obs_video_writer = cv2.VideoWriter(args.save_obs_video, fourcc, args.control_hz, (84, 84))
        print(f"[Debug] Saving observation video to {args.save_obs_video}")

    def center_crop_square(image: np.ndarray) -> np.ndarray:
        """Center crop to square."""
        h, w = image.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return image[y_start:y_start + size, x_start:x_start + size]

    # Frame buffer for stacking low_dim_state
    state_buffer = deque(maxlen=policy.frame_stack)

    # Control timing
    control_dt = 1.0 / args.control_hz

    print("\n" + "=" * 60)
    print("Ready to run. Press Ctrl+C to stop.")
    print("=" * 60)

    # Training initial position (from curriculum_stage=3 in lift_cube.py)
    # Gripper positioned above cube, open, with wrist joints at π/2 (top-down)
    FINGER_WIDTH_OFFSET = -0.015  # Static finger is offset from gripper center
    GRASP_Z_OFFSET = 0.005
    CUBE_Z = 0.015  # Cube height on table

    # Genesis vs MuJoCo reset height difference
    # MuJoCo: starts above cube (HEIGHT_OFFSET = 0.03)
    # Genesis: starts at grasp height (HEIGHT_OFFSET = 0.0)
    if use_genesis:
        HEIGHT_OFFSET = 0.0   # Genesis: at grasp height
        RESET_GRIPPER = 0.3   # Genesis uses partially open gripper
    else:
        HEIGHT_OFFSET = 0.03  # MuJoCo: 3cm above grasp height
        RESET_GRIPPER = 1.0   # Fully open

    # Safe positions
    SAFE_JOINTS = np.zeros(5)  # Extended forward - safe for IK movements
    REST_JOINTS = np.array([-0.2424, -1.8040, 1.6582, 0.7309, -0.0629])  # Folded rest

    # Joint offset correction (from kinematic verification devlog 032/036)
    # elbow_flex (joint 2) reads ~12.5deg more bent than actual physical position
    ELBOW_FLEX_OFFSET_RAD = np.deg2rad(-12.5)  # -12.5deg offset

    def apply_joint_offset(joints: np.ndarray) -> np.ndarray:
        """Apply calibration offset correction to joint readings for accurate FK.

        The elbow_flex sensor has a ~12.5deg bias that causes FK position errors.
        This correction aligns sensor readings with physical joint positions.
        """
        corrected = joints.copy()
        corrected[2] += ELBOW_FLEX_OFFSET_RAD  # elbow_flex correction
        return corrected

    def move_to_initial_pose_with_wrist_lock(robot, ik, target_pos, num_steps=100, dt=0.05):
        """Move robot to target EE position using IK with wrist locked at π/2."""
        for step in range(num_steps):
            current_joints_raw = robot.get_joint_positions_radians()
            # Apply joint offset for accurate FK if in genesis mode
            if use_genesis:
                current_joints = apply_joint_offset(current_joints_raw)
            else:
                current_joints = current_joints_raw.copy()

            # Lock wrist joints at π/2
            current_joints[3] = np.pi / 2
            current_joints[4] = -np.pi / 2
            # Multiple IK iterations for better convergence
            for _ in range(3):
                target_joints = ik.compute_ik(target_pos, current_joints, locked_joints=[3, 4])
                current_joints = target_joints
            # Ensure wrist stays locked
            target_joints[3] = np.pi / 2
            target_joints[4] = -np.pi / 2

            # Undo offset for robot command if in genesis mode
            if use_genesis:
                target_joints[2] -= ELBOW_FLEX_OFFSET_RAD

            robot.send_action(target_joints, RESET_GRIPPER)
            time.sleep(dt)

            # Apply offset for FK accuracy check
            if use_genesis:
                ik.sync_joint_positions(apply_joint_offset(robot.get_joint_positions_radians()))
            else:
                ik.sync_joint_positions(robot.get_joint_positions_radians())
            ee_pos = ik.get_ee_position()
            error = np.linalg.norm(target_pos - ee_pos)
            if error < 0.01:  # Within 1cm
                break

        return ee_pos

    def safe_return():
        """Safe return sequence: lift up first, then go to rest position."""
        print("\nSafe return sequence...")

        # Step 1: Lift up to safe height (keep wrist orientation)
        print("  Lifting to safe height...")
        try:
            current_joints_raw = robot.get_joint_positions_radians()
            if use_genesis:
                ik.sync_joint_positions(apply_joint_offset(current_joints_raw))
            else:
                ik.sync_joint_positions(current_joints_raw)
            current_ee = ik.get_ee_position()
            safe_height_target = current_ee.copy()
            safe_height_target[2] = 0.15  # Lift to 15cm

            for step in range(40):
                current_joints_raw = robot.get_joint_positions_radians()
                if use_genesis:
                    current_joints = apply_joint_offset(current_joints_raw)
                else:
                    current_joints = current_joints_raw.copy()
                current_joints[3] = np.pi / 2
                current_joints[4] = -np.pi / 2
                target_joints = ik.compute_ik(safe_height_target, current_joints, locked_joints=[3, 4])
                target_joints[3] = np.pi / 2
                target_joints[4] = -np.pi / 2

                # Undo offset for robot command
                if use_genesis:
                    target_joints[2] -= ELBOW_FLEX_OFFSET_RAD

                robot.send_action(target_joints, 1.0)
                time.sleep(0.05)

                if use_genesis:
                    ik.sync_joint_positions(apply_joint_offset(robot.get_joint_positions_radians()))
                else:
                    ik.sync_joint_positions(robot.get_joint_positions_radians())
                ee_pos = ik.get_ee_position()
                if ee_pos[2] > 0.12:  # High enough
                    break

            print(f"  Lifted to: {ik.get_ee_position()}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Warning: Failed to lift ({e}), going directly to rest...")

        # Step 2: Return to rest position with gripper closed
        print("  Returning to rest position...")
        robot.send_action(REST_JOINTS, -1.0)  # Close gripper at rest
        time.sleep(1.0)

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
                if camera is not None:
                    wrist_path = record_dir / f"{episode_prefix}_wrist.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    wrist_writer = cv2.VideoWriter(str(wrist_path), fourcc, args.control_hz, (480, 480))
                    print(f"  Recording wrist: {wrist_path}")

                # External camera writer
                if external_cap is not None:
                    ext_w = int(external_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ext_h = int(external_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ext_size = min(ext_w, ext_h)
                    external_path = record_dir / f"{episode_prefix}_external.mp4"
                    external_writer = cv2.VideoWriter(str(external_path), fourcc, args.control_hz, (ext_size, ext_size))
                    print(f"  Recording external: {external_path}")

            # Step 1: Reset to safe extended position
            print("  Step 1: Safe extended position...")
            robot.send_action(SAFE_JOINTS, RESET_GRIPPER)
            time.sleep(1.5)

            # Step 2: Set wrist joints to π/2 for top-down orientation
            print("  Step 2: Setting top-down wrist orientation...")
            topdown_joints = robot.get_joint_positions_radians().copy()
            topdown_joints[3] = np.pi / 2
            topdown_joints[4] = -np.pi / 2  # wrist_roll (flipped for real robot)
            robot.send_action(topdown_joints, RESET_GRIPPER)
            time.sleep(1.0)

            # Step 3: Move to training initial position with wrist locked
            reset_desc = "grasp height" if use_genesis else "above cube"
            print(f"  Step 3: Moving to {reset_desc} position...")
            initial_target = np.array([
                args.cube_x,
                args.cube_y + FINGER_WIDTH_OFFSET,
                CUBE_Z + GRASP_Z_OFFSET + HEIGHT_OFFSET
            ])
            print(f"    Target: {initial_target}")

            ee_pos = move_to_initial_pose_with_wrist_lock(robot, ik, initial_target)
            print(f"    Reached: {ee_pos}")

            # Reset buffers
            state_buffer.clear()
            if camera is not None:
                camera.reset()
                if not camera.fill_buffer():
                    print("  Failed to fill camera buffer, skipping episode")
                    continue

            # Get initial robot state and compute FK
            joint_pos_rad = robot.get_joint_positions_radians()
            joint_vel = robot.get_joint_velocities()
            gripper_state = robot.get_gripper_position()

            # Use IK controller for FK (sync joints, read EE pose)
            # Apply joint offset for accurate FK if in genesis mode
            if use_genesis:
                ik.sync_joint_positions(apply_joint_offset(joint_pos_rad))
            else:
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
            # Track global step for sim frame indexing
            global_step = episode * args.episode_length

            for step in range(args.episode_length):
                step_start = time.time()

                # Get RGB observation
                if use_sim_obs:
                    # Use pre-recorded sim frames for sim2real diagnostic
                    frame_idx = (global_step + step) % len(sim_frames)
                    # For DrQ-v2, need to stack frames
                    rgb_obs = np.stack([
                        sim_frames[max(0, frame_idx - i)]
                        for i in range(policy.frame_stack - 1, -1, -1)
                    ], axis=0)
                    if step == 0:
                        print(f"  [SIM2REAL TEST] Using sim frame {frame_idx}/{len(sim_frames)}")
                elif use_mock_camera:
                    rgb_obs = np.random.randint(
                        0, 256, (policy.frame_stack, 3, 84, 84), dtype=np.uint8
                    )
                else:
                    rgb_obs = camera.get_stacked_observation()
                    if rgb_obs is None:
                        print("  Camera frame error, ending episode")
                        break

                    # Apply image rotation if specified
                    if args.rotate_image != 0:
                        rotated_stack = []
                        for i in range(rgb_obs.shape[0]):
                            img_hwc = np.transpose(rgb_obs[i], (1, 2, 0))
                            if args.rotate_image == 90:
                                img_hwc = cv2.rotate(img_hwc, cv2.ROTATE_90_CLOCKWISE)
                            elif args.rotate_image == 180:
                                img_hwc = cv2.rotate(img_hwc, cv2.ROTATE_180)
                            elif args.rotate_image == 270:
                                img_hwc = cv2.rotate(img_hwc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            rotated_stack.append(np.transpose(img_hwc, (2, 0, 1)))
                        rgb_obs = np.stack(rotated_stack, axis=0)

                    # Save observation for debugging
                    if args.save_obs and step < 5:
                        img_hwc = np.transpose(rgb_obs[-1], (1, 2, 0))
                        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"drq_obs_ep{episode+1}_step{step}.png", img_bgr)
                        print(f"    Saved observation: drq_obs_ep{episode+1}_step{step}.png")

                    # Write to observation video
                    if obs_video_writer is not None:
                        img_hwc = np.transpose(rgb_obs[-1], (1, 2, 0))
                        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                        obs_video_writer.write(img_bgr)

                    # Record wrist camera (get raw frame for higher quality)
                    if wrist_writer is not None:
                        raw_frame = camera.get_raw_frame()
                        if raw_frame is not None:
                            # Center crop to square
                            cropped = center_crop_square(raw_frame)
                            # Resize to 480x480 for reasonable file size
                            resized = cv2.resize(cropped, (480, 480))
                            wrist_writer.write(resized)

                # Record external camera
                if external_writer is not None and external_cap is not None:
                    ret, ext_frame = external_cap.read()
                    if ret:
                        cropped = center_crop_square(ext_frame)
                        external_writer.write(cropped)

                # Stack low_dim states
                if sim_states is not None:
                    # Use pre-recorded sim states for full sim observation playback
                    state_idx = (global_step + step) % len(sim_states)
                    # Need to stack states for DrQ-v2
                    low_dim_obs = np.stack([
                        sim_states[max(0, state_idx - i)]
                        for i in range(policy.frame_stack - 1, -1, -1)
                    ], axis=0).astype(np.float32)
                    if step == 0:
                        print(f"  [SIM2REAL TEST] Using sim state {state_idx}/{len(sim_states)}")
                else:
                    low_dim_obs = np.stack(list(state_buffer), axis=0).astype(np.float32)

                # Debug state output
                if args.debug_state and step == 0:
                    print("\n  [DEBUG] Low-dim state breakdown (step 0):")
                    latest_state = low_dim_obs[-1]
                    print(f"    joint_pos (6): {latest_state[0:6]}")
                    print(f"    joint_vel (6): {latest_state[6:12]}")
                    print(f"    gripper_pos (3): {latest_state[12:15]}")
                    print(f"    gripper_euler (3): {latest_state[15:18]}")
                    if len(latest_state) > 18:
                        print(f"    cube_pos (3): {latest_state[18:21]}")

                # Get action from policy
                action = policy.get_action(rgb_obs, low_dim_obs)

                # Debug action output
                if args.debug_state and step < 10:
                    print(f"    [DEBUG] Step {step}: raw_action={action} (before any transforms)")

                # Parse action
                delta_xyz = action[:3].copy()  # In [-1, 1], will be scaled
                gripper_action = action[3]  # -1 = closed, 1 = open

                # Transform from Genesis frame to MuJoCo frame (90deg rotation)
                if use_genesis:
                    genesis_x, genesis_y, genesis_z = delta_xyz
                    delta_xyz = np.array([
                        -genesis_y,  # MuJoCo X (forward) = -Genesis Y
                        genesis_x,   # MuJoCo Y (sideways) = Genesis X
                        genesis_z,   # Z unchanged
                    ])
                    if args.debug_state and step < 10:
                        print(f"    [DEBUG] Transformed: genesis({genesis_x:.3f},{genesis_y:.3f},{genesis_z:.3f}) -> mujoco({delta_xyz[0]:.3f},{delta_xyz[1]:.3f},{delta_xyz[2]:.3f})")

                # Use IK to convert Cartesian action to joint targets
                current_joints_raw = robot.get_joint_positions_radians()

                # Apply joint offset for accurate IK (Genesis mode)
                if use_genesis:
                    current_joints = apply_joint_offset(current_joints_raw)
                else:
                    current_joints = current_joints_raw.copy()

                target_joints = ik.cartesian_to_joints(
                    delta_xyz,
                    current_joints,
                    action_scale=args.action_scale,
                    locked_joints=[3, 4] if use_genesis else [4],  # Lock both wrist joints for Genesis
                )

                # Convert back from corrected joints to raw joints for robot command
                if use_genesis:
                    target_joints[2] -= ELBOW_FLEX_OFFSET_RAD

                # Debug IK
                if args.debug_state and step < 5:
                    ik.sync_joint_positions(current_joints)
                    current_ee = ik.get_ee_position()
                    target_ee = current_ee + delta_xyz * args.action_scale
                    joint_diff = target_joints - current_joints_raw
                    print(f"    [IK DEBUG] current_ee=[{current_ee[0]:.4f},{current_ee[1]:.4f},{current_ee[2]:.4f}]")
                    print(f"    [IK DEBUG] target_ee=[{target_ee[0]:.4f},{target_ee[1]:.4f},{target_ee[2]:.4f}]")
                    print(f"    [IK DEBUG] joint_diff={joint_diff}")

                # Send action to robot
                if not args.dry_run:
                    robot.send_action(target_joints, gripper_action)

                # Update state for next step
                joint_pos_rad = robot.get_joint_positions_radians()
                joint_vel = robot.get_joint_velocities()
                gripper_state = robot.get_gripper_position()

                # FK for EE state (apply offset for Genesis)
                if use_genesis:
                    ik.sync_joint_positions(apply_joint_offset(joint_pos_rad))
                else:
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

                # Status - show full EE position for Genesis mode
                if step % 20 == 0:
                    if use_genesis:
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
        if 'wrist_writer' in dir() and wrist_writer is not None:
            wrist_writer.release()
        if 'external_writer' in dir() and external_writer is not None:
            external_writer.release()

    finally:
        print("\nCleaning up...")
        if obs_video_writer is not None:
            obs_video_writer.release()
            print(f"  Saved observation video: {args.save_obs_video}")
        if camera is not None:
            camera.close()
        if external_cap is not None:
            external_cap.release()
        safe_return()
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
