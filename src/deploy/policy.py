"""Policy loading and inference for sim-to-real transfer.

Loads trained checkpoints (DrQ-v2 or Genesis PPO) and runs inference.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            frame_stack = self._cfg.frame_stack  # 3
            image_size = self._cfg.env.image_size  # 84
            action_dim = 4  # delta XYZ + gripper

            # Auto-detect state_dim from checkpoint weights
            # low_dim_obs weight shape is (hidden, frame_stack * state_dim)
            agent_state = payload["agent"]
            low_dim_weight_key = "actor_model.input_preprocess_modules.low_dim_obs.0.weight"
            if low_dim_weight_key in agent_state:
                total_low_dim = agent_state[low_dim_weight_key].shape[1]
                state_dim = total_low_dim // frame_stack
                print(f"  Auto-detected state_dim={state_dim} from checkpoint (total={total_low_dim}, frame_stack={frame_stack})")
            else:
                # Fallback to default
                state_dim = 21  # joint_pos(6) + joint_vel(6) + gripper_pos(3) + gripper_euler(3) + cube_pos(3)
                print(f"  Using default state_dim={state_dim}")

            # Store for property access
            self._state_dim = state_dim

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
        """State dimension (auto-detected from checkpoint)."""
        return getattr(self, "_state_dim", 21)

    @property
    def frame_stack(self) -> int:
        """Number of stacked frames."""
        if self._cfg is not None:
            return self._cfg.frame_stack
        return 3


class LowDimStateBuilder:
    """Builds low-dim state vector for policy input.

    The sim training uses a 21-dim state:
    - joint_pos (6): Joint positions in radians (5 arm + 1 gripper)
    - joint_vel (6): Joint velocities (5 arm + 1 gripper)
    - gripper_pos (3): End-effector XYZ position
    - gripper_euler (3): End-effector orientation (roll, pitch, yaw)
    - cube_pos (3): Cube XYZ position

    For real robot deployment, cube_pos is NOT available from sensors.
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
        gripper_state: float = 0.0,
        cube_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build state vector.

        Args:
            joint_pos: Arm joint positions (5,) in radians.
            joint_vel: Arm joint velocities (5,).
            gripper_pos: End-effector position (3,).
            gripper_euler: End-effector orientation as euler angles (3,).
            gripper_state: Gripper joint position (scalar, default 0).
            cube_pos: Cube position (3,). Required if include_cube_pos=True.

        Returns:
            State vector (21,) float32.
        """
        # Ensure joint arrays have 6 elements (5 arm + 1 gripper)
        joint_pos = np.atleast_1d(joint_pos).astype(np.float32)
        joint_vel = np.atleast_1d(joint_vel).astype(np.float32)

        # Pad to 6 joints if only 5 provided
        if len(joint_pos) == 5:
            joint_pos = np.append(joint_pos, gripper_state)
        if len(joint_vel) == 5:
            joint_vel = np.append(joint_vel, 0.0)

        parts = [
            joint_pos[:6],
            joint_vel[:6],
            np.atleast_1d(gripper_pos).astype(np.float32),
            np.atleast_1d(gripper_euler).astype(np.float32),
        ]

        if self.include_cube_pos:
            if cube_pos is None:
                raise ValueError("cube_pos required when include_cube_pos=True")
            parts.append(np.atleast_1d(cube_pos).astype(np.float32))
        else:
            parts.append(np.zeros(3, dtype=np.float32))

        return np.concatenate(parts)

    @property
    def state_dim(self) -> int:
        return self._state_dim


# =============================================================================
# Genesis PPO Model Classes
# =============================================================================


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNEncoder(nn.Module):
    """CNN encoder for image observations (Nature CNN architecture)."""

    def __init__(self, image_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        # Standard nature CNN architecture
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(image_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # For 84x84 input: 64 * 7 * 7 = 3136
        self.fc = layer_init(nn.Linear(3136, feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width), values in [0, 255]
        x = x.float() / 255.0
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x


class ActorCritic(nn.Module):
    """Actor-Critic network with CNN encoder for Genesis PPO."""

    def __init__(
        self,
        image_channels: int = 3,
        low_dim_size: int = 18,
        action_dim: int = 4,
        feature_dim: int = 256,
    ):
        super().__init__()

        self.cnn_encoder = CNNEncoder(image_channels, feature_dim)

        # Low-dim state encoder
        self.low_dim_encoder = nn.Sequential(
            layer_init(nn.Linear(low_dim_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )

        # Combined feature size: CNN features + low-dim features
        combined_dim = feature_dim + 64  # 256 + 64 = 320

        # Actor network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        # Learnable log std
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )


class GenesisPPORunner:
    """Runs inference with a Genesis-trained PPO policy.

    Unlike DrQ-v2, this uses single frames (no frame stacking):
    - rgb: (3, 84, 84) uint8
    - low_dim_state: (18,) float32

    Action output:
    - action: (4,) float32 in [-1, 1]
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ):
        """Initialize PPO policy runner.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            device: Torch device for inference.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._network = None

    def load(self) -> bool:
        """Load checkpoint and initialize network.

        Returns:
            True if loaded successfully.
        """
        if not self.checkpoint_path.exists():
            print(f"Checkpoint not found: {self.checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

            self._network = ActorCritic(
                image_channels=3,
                low_dim_size=18,
                action_dim=4,
                feature_dim=256,
            ).to(self.device)

            self._network.load_state_dict(checkpoint['network_state_dict'])
            self._network.eval()

            print(f"Loaded PPO checkpoint: {self.checkpoint_path}")
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
        """Run inference to get deterministic action.

        Args:
            rgb: Image observation (3, H, W) uint8.
            low_dim_state: Low-dim state (18,) float32.

        Returns:
            Action (4,) float32 in [-1, 1].
        """
        if self._network is None:
            raise RuntimeError("Policy not loaded. Call load() first.")

        with torch.no_grad():
            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
            state_tensor = torch.from_numpy(low_dim_state).unsqueeze(0).float().to(self.device)

            # Forward through encoders
            img_features = self._network.cnn_encoder(rgb_tensor)
            low_dim_features = self._network.low_dim_encoder(state_tensor)
            features = torch.cat([img_features, low_dim_features], dim=-1)

            # Get deterministic action (mean, no sampling)
            action = self._network.actor_mean(features)

            return action.squeeze(0).cpu().numpy()

    @property
    def action_dim(self) -> int:
        """Action dimension (4: delta XYZ + gripper)."""
        return 4

    @property
    def state_dim(self) -> int:
        """State dimension (18: proprioception only)."""
        return 18

    @property
    def frame_stack(self) -> int:
        """Frame stack (1 for PPO, no stacking)."""
        return 1


class SegDepthPolicyRunner:
    """Runs inference with a trained DrQ-v2 policy using seg+depth observations.

    The policy expects:
    - seg_depth: (batch, frame_stack, 2, 84, 84) float32 normalized [0, 1]
    - low_dim_state: (batch, frame_stack, state_dim) float32

    And outputs:
    - action: (batch, action_dim) float32 in [-1, 1]

    Action space (4 dims):
    - delta X, Y, Z for end-effector position
    - gripper open/close (-1 to 1)

    Note: Uses "rgb" as observation key to match training wrapper convention,
    even though the actual input is seg+depth (2 channels instead of 3).
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

            # Import hydra for agent creation
            import hydra

            # Get observation and action space info from config
            frame_stack = self._cfg.frame_stack  # 3
            image_size = self._cfg.env.image_size  # 84
            action_dim = 4  # delta XYZ + gripper

            # Seg+depth: 2 channels per frame, stacked along channel axis
            # With frame_stack=3 and 2 channels: (6, 84, 84)
            obs_channels = 2
            stacked_channels = frame_stack * obs_channels  # 6

            # Auto-detect state_dim from checkpoint weights
            agent_state = payload["agent"]
            low_dim_weight_key = "actor_model.input_preprocess_modules.low_dim_obs.0.weight"
            if low_dim_weight_key in agent_state:
                total_low_dim = agent_state[low_dim_weight_key].shape[1]
                state_dim = total_low_dim // frame_stack
                print(
                    f"  Auto-detected state_dim={state_dim} from checkpoint (total={total_low_dim}, frame_stack={frame_stack})"
                )
            else:
                # Fallback to proprioception only (no cube_pos for sim2real)
                state_dim = 18
                print(f"  Using default state_dim={state_dim}")

            # Store for property access
            self._state_dim = state_dim

            # Observation space with frame_stack_on_channel=True
            # Encoder expects (num_views, channels, height, width): (1, 6, 84, 84)
            observation_space = {
                "rgb": {"shape": (1, stacked_channels, image_size, image_size)},
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

            print(f"Loaded seg+depth checkpoint: {self.checkpoint_path}")
            return True

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_action(
        self,
        seg_depth: np.ndarray,
        low_dim_state: np.ndarray,
    ) -> np.ndarray:
        """Run inference to get action.

        Args:
            seg_depth: Seg+depth observation (6, 84, 84) uint8 - frame_stack * 2 channels.
            low_dim_state: Low-dim state (frame_stack, state_dim) float32.

        Returns:
            Action (action_dim,) float32 in [-1, 1].
        """
        if self._agent is None:
            raise RuntimeError("Policy not loaded. Call load() first.")

        with torch.no_grad():
            # Normalize to [0, 1] as in training
            seg_depth_norm = seg_depth.astype(np.float32) / 255.0

            # Add batch and view dimensions: (C, H, W) -> (1, 1, C, H, W)
            # batch=1, num_views=1, channels=6, height=84, width=84
            obs_tensor = torch.from_numpy(seg_depth_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            state_tensor = (
                torch.from_numpy(low_dim_state).unsqueeze(0).float().to(self.device)
            )

            obs = {
                "rgb": obs_tensor,  # Key is "rgb" to match training wrapper
                "low_dim_state": state_tensor,
            }

            # Get action from agent
            action = self._agent.act(obs, step=self._step, eval_mode=True)

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Remove batch dim: (1, action_dim) -> (action_dim,)
            action = action.squeeze(0)

            # Handle ActionSequence wrapper output if present
            if action.ndim > 1:
                action = action[0]

            return action

    @property
    def action_dim(self) -> int:
        """Action dimension (4: delta XYZ + gripper)."""
        return 4

    @property
    def state_dim(self) -> int:
        """State dimension (auto-detected from checkpoint)."""
        return getattr(self, "_state_dim", 18)

    @property
    def frame_stack(self) -> int:
        """Number of stacked frames."""
        if self._cfg is not None:
            return self._cfg.frame_stack
        return 3
