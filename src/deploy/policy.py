"""Policy loading and inference for sim-to-real transfer.

Loads a trained DrQ-v2 checkpoint and runs inference.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# Add pick-101 paths for robobase and training modules
pick101_root = Path("/home/gota/ggando/ml/pick-101")
if str(pick101_root) not in sys.path:
    sys.path.insert(0, str(pick101_root))
if str(pick101_root / "external" / "robobase") not in sys.path:
    sys.path.insert(0, str(pick101_root / "external" / "robobase"))


class PolicyRunner:
    """Runs inference with a trained DrQ-v2 policy.

    The policy expects:
    - rgb: (batch, frame_stack, 3, 84, 84) float32
    - low_dim_state: (batch, frame_stack, state_dim) float32

    And outputs:
    - action: (batch, action_dim) float32 in [-1, 1]

    Action space (4 dims):
    - delta X, Y, Z for end-effector position (scaled by 0.02 = 2cm/step in sim)
    - gripper open/close (-1 to 1)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ):
        """Initialize policy runner.

        Args:
            checkpoint_path: Path to snapshot .pt file.
            device: Torch device for inference.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        self._agent = None
        self._cfg = None
        self._step = 0  # Used for exploration schedule (1e6 = eval mode)

    def load(self) -> bool:
        """Load checkpoint and initialize agent.

        Returns:
            True if loaded successfully.
        """
        if not self.checkpoint_path.exists():
            print(f"Checkpoint not found: {self.checkpoint_path}")
            return False

        try:
            # Load checkpoint
            with open(self.checkpoint_path, "rb") as f:
                payload = torch.load(f, map_location="cpu", weights_only=False)

            self._cfg = payload["cfg"]

            # Import hydra and robobase for agent creation
            import hydra

            # Get observation and action space info from config
            # These are hardcoded based on the training setup
            # TODO: Save these in the checkpoint for portability
            frame_stack = self._cfg.frame_stack  # 3
            image_size = self._cfg.env.image_size  # 84
            state_dim = 21  # joint_pos(6) + joint_vel(6) + gripper_pos(3) + gripper_euler(3) + cube_pos(3)
            action_dim = 4  # delta XYZ + gripper

            observation_space = {
                "rgb": {"shape": (frame_stack, 3, image_size, image_size)},
                "low_dim_state": {"shape": (frame_stack, state_dim)},
            }
            action_space = {
                "shape": (action_dim,),
                "minimum": -1.0,
                "maximum": 1.0,
            }

            # Create agent using hydra
            self._agent = hydra.utils.instantiate(
                self._cfg.method,
                device=self.device,
                observation_space=observation_space,
                action_space=action_space,
                num_train_envs=self._cfg.num_train_envs,
                replay_alpha=self._cfg.replay.alpha,
                replay_beta=self._cfg.replay.beta,
                frame_stack_on_channel=self._cfg.frame_stack_on_channel,
                intrinsic_reward_module=None,
            )

            # Load weights
            self._agent.load_state_dict(payload["agent"])
            self._agent.train(False)

            # Set step high for eval mode (no exploration noise)
            self._step = 1_000_000

            print(f"Loaded checkpoint: {self.checkpoint_path}")
            return True

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_action(
        self,
        rgb: np.ndarray,
        low_dim_state: np.ndarray,
    ) -> np.ndarray:
        """Run inference to get action.

        Args:
            rgb: Image observation (frame_stack, 3, H, W) uint8.
            low_dim_state: Low-dim state (frame_stack, state_dim) float32.

        Returns:
            Action (action_dim,) float32 in [-1, 1].
        """
        if self._agent is None:
            raise RuntimeError("Policy not loaded. Call load() first.")

        with torch.no_grad():
            # Prepare observation tensors
            # Add batch dimension: (F, C, H, W) -> (1, F, C, H, W)
            rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).float().to(self.device)
            state_tensor = torch.from_numpy(low_dim_state).unsqueeze(0).float().to(self.device)

            obs = {
                "rgb": rgb_tensor,
                "low_dim_state": state_tensor,
            }

            # Get action from agent
            action = self._agent.act(obs, step=self._step, eval_mode=True)

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Remove batch dim: (1, action_dim) -> (action_dim,)
            action = action.squeeze(0)

            # Handle ActionSequence wrapper output if present
            # (seq, action_dim) -> (action_dim,) take first
            if action.ndim > 1:
                action = action[0]

            return action

    @property
    def action_dim(self) -> int:
        """Action dimension (4: delta XYZ + gripper)."""
        return 4

    @property
    def state_dim(self) -> int:
        """State dimension (21 in sim, may differ for real)."""
        return 21

    @property
    def frame_stack(self) -> int:
        """Number of stacked frames."""
        if self._cfg is not None:
            return self._cfg.frame_stack
        return 3


class LowDimStateBuilder:
    """Builds low-dim state vector for policy input.

    The sim training uses a 21-dim state:
    - joint_pos (6): Joint positions in radians
    - joint_vel (6): Joint velocities
    - gripper_pos (3): End-effector XYZ position
    - gripper_euler (3): End-effector orientation (roll, pitch, yaw)
    - cube_pos (3): Cube XYZ position

    For real robot deployment, cube_pos is NOT available from sensors.
    Options:
    1. Zero out cube_pos (may degrade performance)
    2. Use vision to estimate cube_pos
    3. Retrain without cube_pos

    This class helps construct the state vector from real robot sensors.
    """

    def __init__(self, include_cube_pos: bool = False):
        """Initialize state builder.

        Args:
            include_cube_pos: If True, expect cube_pos in build().
                             If False, zero-pad those dimensions.
        """
        self.include_cube_pos = include_cube_pos
        self._state_dim = 21

    def build(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        gripper_pos: np.ndarray,
        gripper_euler: np.ndarray,
        cube_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build state vector.

        Args:
            joint_pos: Joint positions (6,).
            joint_vel: Joint velocities (6,).
            gripper_pos: End-effector position (3,).
            gripper_euler: End-effector orientation as euler angles (3,).
            cube_pos: Cube position (3,). Required if include_cube_pos=True.

        Returns:
            State vector (21,) float32.
        """
        # Ensure all inputs are 1D arrays
        parts = [
            np.atleast_1d(joint_pos).astype(np.float32),
            np.atleast_1d(joint_vel).astype(np.float32),
            np.atleast_1d(gripper_pos).astype(np.float32),
            np.atleast_1d(gripper_euler).astype(np.float32),
        ]

        if self.include_cube_pos:
            if cube_pos is None:
                raise ValueError("cube_pos required when include_cube_pos=True")
            parts.append(np.atleast_1d(cube_pos).astype(np.float32))
        else:
            # Zero-pad cube_pos dimensions
            parts.append(np.zeros(3, dtype=np.float32))

        return np.concatenate(parts)

    @property
    def state_dim(self) -> int:
        return self._state_dim
