#!/usr/bin/env python3
"""Real robot inference with Genesis PPO policy.

Runs a trained Genesis PPO policy on the real SO-101 robot.
Unlike DrQ-v2, this uses single frames (no frame stacking).

Usage:
    # Dry run (mock robot and camera)
    uv run python scripts/ppo_inference.py --checkpoint /path/to/checkpoint.pt --dry_run

    # Real robot
    uv run python scripts/ppo_inference.py --checkpoint /path/to/checkpoint.pt

    # With video recording (wrist + external camera)
    uv run python scripts/ppo_inference.py --checkpoint /path/to/checkpoint.pt \
        --record_dir ./recordings --external_camera 2

Architecture:
    Camera → Preprocess → Single Frame ─┐
                                        ├─→ Policy → Cartesian Action → IK → Joint Commands → Robot
    Robot State → FK → low_dim_state ───┘

Note: PPO uses single frames (no stacking), unlike DrQ-v2 which uses 3-frame stacks.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import local deploy modules
from src.deploy.camera import CameraPreprocessor
from src.deploy.policy import GenesisPPORunner, LowDimStateBuilder
from src.deploy.robot import SO101Robot, MockSO101Robot
from src.deploy.controllers import IKController


def main():
    parser = argparse.ArgumentParser(description="Run Genesis PPO policy on real SO-101 robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PPO checkpoint (.pt file)",
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
    parser.add_argument(
        "--flip_y",
        action="store_true",
        help="Flip Y-axis of actions (for coordinate system mismatch debugging)",
    )
    parser.add_argument(
        "--save_obs",
        action="store_true",
        help="Save observation images to debug visual mismatch",
    )
    parser.add_argument(
        "--rotate_image",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate camera image by degrees (for orientation mismatch)",
    )
    parser.add_argument(
        "--save_obs_video",
        type=str,
        default=None,
        help="Save observation images as video (path to output .mp4)",
    )
    parser.add_argument(
        "--debug_state",
        action="store_true",
        help="Print verbose state/action debug info at each step",
    )
    parser.add_argument(
        "--genesis_wrist",
        action="store_true",
        help="Use Genesis wrist configuration (joint[4]=+π/2 instead of -π/2)",
    )
    parser.add_argument(
        "--fix_euler",
        action="store_true",
        help="Transform euler angles to match Genesis (compensate for wrist_roll π difference)",
    )
    parser.add_argument(
        "--sim_frames_dir",
        type=str,
        default=None,
        help="Directory with pre-recorded sim frames (PNG/NPY) to use instead of real camera. "
             "If policy works with sim frames but not real frames, proves visual sim2real gap.",
    )
    parser.add_argument(
        "--sim_states_file",
        type=str,
        default=None,
        help="NPY file with sim low_dim_states (N, 18) to use instead of real robot state. "
             "Use with --sim_frames_dir for full sim observation playback.",
    )
    parser.add_argument(
        "--genesis_to_mujoco",
        action="store_true",
        help="Transform actions from Genesis frame to MuJoCo frame. "
             "Genesis: X=sideways, -Y=forward, Z=up. "
             "MuJoCo: X=forward, Y=sideways, Z=up. "
             "Transformation: MuJoCo_X=-Genesis_Y, MuJoCo_Y=Genesis_X",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SO-101 Genesis PPO Inference (Single Frame)")
    print("=" * 60)

    # === 1. Load Policy ===
    print("\n[1/4] Loading PPO policy...")
    policy = GenesisPPORunner(args.checkpoint, device=args.device)
    if not policy.load():
        print("Failed to load policy. Exiting.")
        return
    print(f"  PPO policy loaded (single frame, no stacking)")
    if args.genesis_to_mujoco:
        print(f"  [FRAME TRANSFORM] Genesis→MuJoCo coordinate transformation enabled")
        print(f"    Genesis: X=sideways, -Y=forward, Z=up")
        print(f"    MuJoCo:  X=forward, Y=sideways, Z=up")

    # === Load sim frames if provided (for sim2real diagnostic) ===
    sim_frames = None
    sim_states = None
    if args.sim_frames_dir:
        sim_frames_dir = Path(args.sim_frames_dir)
        if not sim_frames_dir.exists():
            print(f"ERROR: Sim frames directory not found: {sim_frames_dir}")
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
                return

            print(f"  Loading {len(frame_files)} PNG files...")
            sim_frames = []
            for f in frame_files:
                img_bgr = cv2.imread(str(f))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Resize to 84x84 if needed
                if img_rgb.shape[:2] != (84, 84):
                    img_rgb = cv2.resize(img_rgb, (84, 84), interpolation=cv2.INTER_AREA)
                # HWC -> CHW
                frame = np.transpose(img_rgb, (2, 0, 1)).astype(np.uint8)
                sim_frames.append(frame)
            sim_frames = np.array(sim_frames)
            print(f"  Loaded frames shape: {sim_frames.shape}")

    if args.sim_states_file:
        sim_states_path = Path(args.sim_states_file)
        if not sim_states_path.exists():
            print(f"ERROR: Sim states file not found: {sim_states_path}")
            return
        sim_states = np.load(sim_states_path)
        print(f"  Loaded sim states shape: {sim_states.shape}")

    use_sim_obs = sim_frames is not None

    # === 2. Initialize Camera ===
    print("\n[2/4] Initializing camera...")
    camera = None
    use_mock_camera = args.dry_run

    if not use_mock_camera:
        # PPO uses single frames (frame_stack=1)
        camera = CameraPreprocessor(
            target_size=(84, 84),
            frame_stack=1,  # Single frame for PPO
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
    state_builder = LowDimStateBuilder()

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

    def center_crop_square(image: np.ndarray) -> np.ndarray:
        """Center crop to square."""
        h, w = image.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return image[y_start:y_start + size, x_start:x_start + size]

    # Control timing
    control_dt = 1.0 / args.control_hz

    # Observation video writer
    obs_video_writer = None
    if args.save_obs_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        obs_video_writer = cv2.VideoWriter(args.save_obs_video, fourcc, args.control_hz, (84, 84))
        print(f"[Debug] Saving observation video to {args.save_obs_video}")

    print("\n" + "=" * 60)
    print("Ready to run. Press Ctrl+C to stop.")
    print("=" * 60)

    # Training initial position (from curriculum_stage=3 in Genesis lift_cube_env.py)
    # Genesis training ends at GRASP HEIGHT (not above cube)
    # See _reset_gripper_near_cube(): Phase 3 moves to grasp_target = cube_z + grasp_z_offset
    FINGER_WIDTH_OFFSET = -0.015  # Static finger is offset from gripper center
    GRASP_Z_OFFSET = 0.005        # Grasp point slightly above cube center
    CUBE_Z = 0.015                # Cube height on table
    # Genesis reset target: (0.25, -0.015, 0.02) = at grasp height, NOT above cube
    RESET_GRIPPER = 0.3           # Genesis uses 0.3 (partially open) during reset, not 1.0

    # Joint offset correction (from kinematic verification devlog 032)
    # elbow_flex (joint 2) reads ~12.5° more bent than actual physical position
    ELBOW_FLEX_OFFSET_RAD = np.deg2rad(-12.5)  # -12.5° offset

    # Safe positions
    SAFE_JOINTS = np.zeros(5)  # Extended forward - safe for IK movements
    REST_JOINTS = np.array([-0.2424, -1.8040, 1.6582, 0.7309, -0.0629])  # Folded rest

    def apply_joint_offset(joints):
        """Apply calibration offset correction to joint readings for accurate FK."""
        corrected = joints.copy()
        corrected[2] += ELBOW_FLEX_OFFSET_RAD  # elbow_flex correction
        return corrected

    def move_to_initial_pose_with_wrist_lock(robot, ik, target_pos, use_genesis_wrist=False, num_steps=100, dt=0.05):
        """Move robot to target EE position using IK with wrist locked at π/2."""
        wrist_roll = np.pi / 2 if use_genesis_wrist else -np.pi / 2
        for step in range(num_steps):
            current_joints = robot.get_joint_positions_radians()
            # Lock wrist joints at π/2
            current_joints[3] = np.pi / 2
            current_joints[4] = wrist_roll
            # Multiple IK iterations for better convergence
            for _ in range(3):
                target_joints = ik.compute_ik(target_pos, current_joints, locked_joints=[3, 4])
                current_joints = target_joints
            # Ensure wrist stays locked
            target_joints[3] = np.pi / 2
            target_joints[4] = wrist_roll
            robot.send_action(target_joints, RESET_GRIPPER)  # Partially open (Genesis uses 0.3)
            time.sleep(dt)

            ik.sync_joint_positions(apply_joint_offset(robot.get_joint_positions_radians()))
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
            current_joints = robot.get_joint_positions_radians()
            ik.sync_joint_positions(apply_joint_offset(current_joints))
            current_ee = ik.get_ee_position()
            safe_height_target = current_ee.copy()
            safe_height_target[2] = 0.15  # Lift to 15cm

            for step in range(40):
                current_joints = robot.get_joint_positions_radians()
                current_joints[3] = np.pi / 2
                current_joints[4] = -np.pi / 2
                target_joints = ik.compute_ik(safe_height_target, current_joints, locked_joints=[3, 4])
                target_joints[3] = np.pi / 2
                target_joints[4] = -np.pi / 2
                robot.send_action(target_joints, 1.0)
                time.sleep(0.05)

                ik.sync_joint_positions(apply_joint_offset(robot.get_joint_positions_radians()))
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
            robot.send_action(SAFE_JOINTS, RESET_GRIPPER)  # Partially open (Genesis uses 0.3)
            time.sleep(1.5)

            # Step 2: Set wrist joints to π/2 for top-down orientation
            print("  Step 2: Setting top-down wrist orientation...")
            topdown_joints = robot.get_joint_positions_radians().copy()
            topdown_joints[3] = np.pi / 2
            # Genesis uses +π/2 for wrist_roll, real robot calibration uses -π/2
            if args.genesis_wrist:
                topdown_joints[4] = np.pi / 2   # Genesis configuration
                print("    Using Genesis wrist config: joint[4]=+π/2")
            else:
                topdown_joints[4] = -np.pi / 2  # Real robot default
            robot.send_action(topdown_joints, RESET_GRIPPER)  # Partially open (Genesis uses 0.3)
            time.sleep(1.0)

            # Step 3: Move to training initial position (at grasp height) with wrist locked
            # Genesis training starts at grasp height, not above cube
            print("  Step 3: Moving to grasp height position...")
            initial_target = np.array([
                args.cube_x,
                args.cube_y + FINGER_WIDTH_OFFSET,
                CUBE_Z + GRASP_Z_OFFSET  # At grasp height (0.02), not above (0.05)
            ])
            print(f"    Target: {initial_target}")

            ee_pos = move_to_initial_pose_with_wrist_lock(robot, ik, initial_target, use_genesis_wrist=args.genesis_wrist)
            print(f"    Reached: {ee_pos}")

            # Reset camera buffer (for PPO, just warm up since no stacking needed)
            if camera is not None:
                camera.reset()
                # Capture a few frames to stabilize
                for _ in range(3):
                    camera.capture_and_preprocess()

            # Get initial robot state and compute FK
            joint_pos_rad = robot.get_joint_positions_radians()
            joint_vel = robot.get_joint_velocities()
            gripper_state = robot.get_gripper_position()

            # Use IK controller for FK (sync joints, read EE pose)
            # Apply joint offset for accurate FK
            ik.sync_joint_positions(apply_joint_offset(joint_pos_rad))
            ee_pos = ik.get_ee_position()
            ee_euler = ik.get_ee_euler()

            print(f"  Initial EE position: {ee_pos}")
            print(f"  Initial joints (rad): {joint_pos_rad}")

            # Episode loop (no frame buffer needed for PPO - single frame)
            # Track global step for sim frame indexing
            global_step = episode * args.episode_length

            for step in range(args.episode_length):
                step_start = time.time()

                # Get single RGB frame (no stacking for PPO)
                if use_sim_obs:
                    # Use pre-recorded sim frames for sim2real diagnostic
                    frame_idx = (global_step + step) % len(sim_frames)
                    rgb_obs = sim_frames[frame_idx].copy()
                    if step == 0:
                        print(f"  [SIM2REAL TEST] Using sim frame {frame_idx}/{len(sim_frames)}")
                elif use_mock_camera:
                    rgb_obs = np.random.randint(0, 256, (3, 84, 84), dtype=np.uint8)
                else:
                    rgb_obs = camera.capture_and_preprocess()
                    if rgb_obs is None:
                        print("  Camera frame error, ending episode")
                        break

                    # Apply image rotation if specified (for camera orientation mismatch)
                    if args.rotate_image != 0:
                        # CHW -> HWC for rotation
                        img_hwc = np.transpose(rgb_obs, (1, 2, 0))
                        if args.rotate_image == 90:
                            img_hwc = cv2.rotate(img_hwc, cv2.ROTATE_90_CLOCKWISE)
                        elif args.rotate_image == 180:
                            img_hwc = cv2.rotate(img_hwc, cv2.ROTATE_180)
                        elif args.rotate_image == 270:
                            img_hwc = cv2.rotate(img_hwc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        # HWC -> CHW
                        rgb_obs = np.transpose(img_hwc, (2, 0, 1))

                    # Save observation image for debugging visual mismatch
                    if args.save_obs and step < 5:
                        # Convert CHW to HWC for saving
                        img_hwc = np.transpose(rgb_obs, (1, 2, 0))
                        # Convert RGB to BGR for cv2
                        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"ppo_obs_ep{episode+1}_step{step}.png", img_bgr)
                        print(f"    Saved observation: ppo_obs_ep{episode+1}_step{step}.png")

                    # Write to observation video
                    if obs_video_writer is not None:
                        img_hwc = np.transpose(rgb_obs, (1, 2, 0))
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

                # Get single low_dim state (no stacking for PPO)
                # PPO expects 18 dims: joint_pos(6) + joint_vel(6) + gripper_pos(3) + gripper_euler(3)
                # LowDimStateBuilder outputs 21 dims (includes 3 zeros for cube_pos), so slice to 18

                if sim_states is not None:
                    # Use pre-recorded sim states for full sim observation playback
                    state_idx = (global_step + step) % len(sim_states)
                    low_dim_obs = sim_states[state_idx].astype(np.float32)
                    if step == 0:
                        print(f"  [SIM2REAL TEST] Using sim state {state_idx}/{len(sim_states)}")
                else:
                    # Transform euler angles if needed to compensate for wrist_roll π difference
                    euler_for_policy = ee_euler.copy()
                    if args.fix_euler:
                        # Genesis uses joint[4]=+π/2, real robot uses -π/2
                        # This π difference affects the yaw component
                        # Transform: wrap yaw by subtracting π (or adding, depending on direction)
                        euler_for_policy[2] = ee_euler[2] - np.pi  # Adjust yaw
                        # Also flip the roll sign to account for flipped frame
                        euler_for_policy[0] = -ee_euler[0]
                        # Normalize to [-π, π]
                        euler_for_policy[2] = np.arctan2(np.sin(euler_for_policy[2]), np.cos(euler_for_policy[2]))

                    low_dim_obs = state_builder.build(
                        joint_pos=joint_pos_rad,
                        joint_vel=joint_vel,
                        gripper_pos=ee_pos,
                        gripper_euler=euler_for_policy,
                        gripper_state=gripper_state,
                    ).astype(np.float32)[:18]  # PPO trained with 18-dim state (no cube_pos)

                # Debug state output
                if args.debug_state and step == 0:
                    print("\n  [DEBUG] Low-dim state breakdown (step 0):")
                    print(f"    joint_pos (6): {low_dim_obs[0:6]}")
                    print(f"    joint_vel (6): {low_dim_obs[6:12]}")
                    print(f"    gripper_pos (3): {low_dim_obs[12:15]}")
                    print(f"    gripper_euler (3): {low_dim_obs[15:18]}")
                    print(f"    Full state vector: {low_dim_obs}")
                    print("\n  [DEBUG] Genesis expected values at reset:")
                    print(f"    gripper_pos: ~[0.25, -0.015, 0.02] (at grasp height)")
                    print(f"    wrist joints: joint[3]=π/2, joint[4]=π/2 (Genesis)")
                    print(f"    Real robot:   joint[3]=π/2, joint[4]=-π/2 (inverted!)")
                    print(f"    This inversion may cause euler angle mismatch!")

                # Get action from policy (single frame input)
                action = policy.get_action(rgb_obs, low_dim_obs)

                # Debug action output
                if args.debug_state and step < 10:
                    print(f"    [DEBUG] Step {step}: raw_action={action} (before clipping)")

                # Clip action to [-1, 1] for safety
                action = np.clip(action, -1.0, 1.0)

                # Parse action
                delta_xyz = action[:3].copy()  # In [-1, 1], will be scaled
                gripper_action = action[3]  # -1 = closed, 1 = open

                # Flip Y if coordinate system mismatch (simple flip)
                if args.flip_y:
                    delta_xyz[1] = -delta_xyz[1]

                # Transform from Genesis frame to MuJoCo frame (90° rotation)
                # Genesis: X=sideways, -Y=forward, Z=up (robot at y=0.314 facing -Y toward y=0)
                # MuJoCo:  X=forward, Y=sideways, Z=up (robot at origin facing +X)
                if args.genesis_to_mujoco:
                    genesis_x, genesis_y, genesis_z = delta_xyz
                    delta_xyz = np.array([
                        -genesis_y,  # MuJoCo X (forward) = -Genesis Y (forward in Genesis is -Y)
                        genesis_x,   # MuJoCo Y (sideways) = Genesis X
                        genesis_z,   # Z unchanged
                    ])
                    if args.debug_state and step < 10:
                        print(f"    [DEBUG] Transformed: genesis({genesis_x:.3f},{genesis_y:.3f},{genesis_z:.3f}) → mujoco({delta_xyz[0]:.3f},{delta_xyz[1]:.3f},{delta_xyz[2]:.3f})")

                # Use IK to convert Cartesian action to joint targets
                current_joints_raw = robot.get_joint_positions_radians()
                # Apply joint offset for accurate IK (sensor reads ~12.5° off on elbow)
                current_joints = apply_joint_offset(current_joints_raw)

                # Debug: show IK target position
                ik.sync_joint_positions(current_joints)
                current_ee = ik.get_ee_position()
                target_ee = current_ee + delta_xyz * args.action_scale

                target_joints = ik.cartesian_to_joints(
                    delta_xyz,
                    current_joints,
                    action_scale=args.action_scale,
                    locked_joints=[3, 4],  # Lock wrist joints for stability (same as reset)
                )

                # Convert back from corrected joints to raw joints for robot command
                # (undo the offset so robot receives correct sensor-space targets)
                target_joints[2] -= ELBOW_FLEX_OFFSET_RAD

                # Debug: show joint changes
                if args.debug_state and step < 5:
                    joint_diff = target_joints - current_joints_raw  # Compare raw to raw
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

                # FK for EE state (apply offset for accurate FK)
                ik.sync_joint_positions(apply_joint_offset(joint_pos_rad))
                ee_pos = ik.get_ee_position()
                ee_euler = ik.get_ee_euler()

                # Status - show full EE position to track motion
                if step % 10 == 0:
                    print(
                        f"  Step {step}: delta=[{delta_xyz[0]:.2f},{delta_xyz[1]:.2f},{delta_xyz[2]:.2f}] "
                        f"gripper={gripper_action:.2f} ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]"
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
