#!/usr/bin/env python3
"""
Test script for randomized IK reset motion.

Runs multiple resets to verify the arm moves to different random positions
within the configured bounds each episode.

Usage:
    python scripts/test_random_ik_reset.py
    python scripts/test_random_ik_reset.py --num-resets 10
    python scripts/test_random_ik_reset.py --config configs/grasp_only_hilserl_train_config.json
"""

import argparse
import json
import logging
import time
import numpy as np
import draccus

# Import robot/teleop modules to register them with draccus
from lerobot.robots import so101_follower  # noqa: F401
from lerobot.teleoperators import so101_leader  # noqa: F401
from lerobot.cameras import opencv  # noqa: F401

from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.envs.configs import HILSerlRobotEnvConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load env config from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Get env config dict and remove 'type' field (handled by draccus registry)
    env_dict = config_dict["env"].copy()
    env_dict.pop("type", None)

    # Decode the env config
    env_cfg = draccus.decode(HILSerlRobotEnvConfig, env_dict)
    return env_cfg


def main():
    parser = argparse.ArgumentParser(description="Test randomized IK reset motion")
    parser.add_argument("--config", type=str,
                        default="configs/grasp_only_hilserl_train_config.json",
                        help="Path to config file with random_ee_reset enabled")
    parser.add_argument("--num-resets", type=int, default=5, help="Number of resets to perform")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause between resets (seconds)")
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    env_cfg = load_config(args.config)

    # Get random reset params from config
    wrapper = env_cfg.wrapper
    base_pos = getattr(wrapper, 'ik_reset_ee_pos', [0.25, 0.0, 0.05])
    range_xy = getattr(wrapper, 'random_ee_range_xy', 0.03)
    range_z = getattr(wrapper, 'random_ee_range_z', 0.02)
    random_enabled = getattr(wrapper, 'random_ee_reset', False)

    logger.info("=" * 60)
    logger.info("Random IK Reset Test")
    logger.info("=" * 60)
    logger.info(f"Random EE reset enabled: {random_enabled}")
    logger.info(f"Base EE position: {base_pos}")
    logger.info(f"Random range XY: ±{range_xy}m (±{range_xy*100}cm)")
    logger.info(f"Random range Z: ±{range_z}m (±{range_z*100}cm)")
    logger.info(f"Number of resets: {args.num_resets}")
    logger.info("=" * 60)

    if not random_enabled:
        logger.warning("random_ee_reset is FALSE in config! Positions will NOT vary.")

    # Create environment
    logger.info("Creating environment...")
    env = make_robot_env(env_cfg)

    # Track reset positions
    reset_positions = []

    try:
        for i in range(args.num_resets):
            logger.info(f"\n{'='*40}")
            logger.info(f"Reset {i+1}/{args.num_resets}")
            logger.info(f"{'='*40}")

            # Perform reset
            obs, info = env.reset()

            # Get current EE position from robot
            robot = env.unwrapped.robot
            if hasattr(robot, '_get_ee_position'):
                # Sync MuJoCo with current joint positions
                current_pos_dict = robot.bus.sync_read("Present_Position", num_retry=3)
                motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
                current_joints_deg = np.array([current_pos_dict[name] for name in motor_names])
                current_joints_rad = np.deg2rad(current_joints_deg)
                robot._sync_mujoco(current_joints_rad)
                current_ee = robot._get_ee_position()

                reset_positions.append(current_ee.copy())
                logger.info(f"Actual EE position: [{current_ee[0]:.4f}, {current_ee[1]:.4f}, {current_ee[2]:.4f}]")

            # Pause to observe
            if i < args.num_resets - 1:
                logger.info(f"Pausing {args.pause}s before next reset...")
                time.sleep(args.pause)

        # Summary statistics
        if reset_positions:
            positions = np.array(reset_positions)
            logger.info(f"\n{'='*60}")
            logger.info("SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Number of resets: {len(positions)}")
            logger.info(f"Base position: {base_pos}")
            logger.info(f"\nActual position statistics:")
            logger.info(f"  X: min={positions[:,0].min():.4f}, max={positions[:,0].max():.4f}, "
                       f"range={positions[:,0].max()-positions[:,0].min():.4f}m")
            logger.info(f"  Y: min={positions[:,1].min():.4f}, max={positions[:,1].max():.4f}, "
                       f"range={positions[:,1].max()-positions[:,1].min():.4f}m")
            logger.info(f"  Z: min={positions[:,2].min():.4f}, max={positions[:,2].max():.4f}, "
                       f"range={positions[:,2].max()-positions[:,2].min():.4f}m")

            # Check if positions are actually different
            if len(positions) > 1:
                distances = []
                for j in range(len(positions) - 1):
                    dist = np.linalg.norm(positions[j+1] - positions[j])
                    distances.append(dist)
                logger.info(f"\nDistance between consecutive resets:")
                logger.info(f"  Mean: {np.mean(distances):.4f}m ({np.mean(distances)*100:.2f}cm)")
                logger.info(f"  Min: {np.min(distances):.4f}m ({np.min(distances)*100:.2f}cm)")
                logger.info(f"  Max: {np.max(distances):.4f}m ({np.max(distances)*100:.2f}cm)")

                if np.mean(distances) < 0.005:
                    logger.warning("WARNING: Positions are very similar - random reset may not be working!")
                else:
                    logger.info("\nSUCCESS: Positions vary between resets as expected")

            logger.info(f"{'='*60}")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")

    finally:
        logger.info("Closing environment...")
        env.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
