"""IK Controller for SO-101 using MuJoCo as kinematics solver.

Uses MuJoCo's Jacobian computation for damped least-squares IK.
Works with real robot by syncing joint positions from sensors.
"""

from pathlib import Path

import mujoco
import numpy as np


class IKController:
    """Damped least-squares IK controller using MuJoCo kinematics.

    For real robot deployment:
    1. Load SO-101 MuJoCo model (for kinematics only, no simulation)
    2. Sync joint positions from real robot sensors
    3. Compute Jacobian and IK to get target joint positions
    4. Send target positions to real robot

    This uses MuJoCo purely as a kinematics solver, not simulator.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        end_effector_site: str = "gripperframe",
        damping: float = 0.1,
        max_dq: float = 0.5,
    ):
        """Initialize IK controller.

        Args:
            model_path: Path to MuJoCo XML model. If None, uses default SO-101 model.
            end_effector_site: Name of the end-effector site in the model.
            damping: Damping factor for singularity robustness.
            max_dq: Maximum joint velocity per step.
        """
        self.damping = damping
        self.max_dq = max_dq

        # Load MuJoCo model
        if model_path is None:
            # Try to find SO-101 model in pick-101 or local models
            possible_paths = [
                Path("/home/gota/ggando/ml/pick-101/models/so101/lift_cube.xml"),
                Path(__file__).parent.parent.parent.parent / "models/so101/lift_cube.xml",
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = p
                    break
            if model_path is None:
                raise FileNotFoundError("Could not find SO-101 MuJoCo model")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Get end-effector site ID
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site
        )
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{end_effector_site}' not found in model")

        # SO-101: 5 arm joints + 1 gripper
        self.n_arm_joints = 5
        self.n_total_joints = self.model.nv

        # Pre-allocate Jacobians
        self.jacp = np.zeros((3, self.n_total_joints))
        self.jacr = np.zeros((3, self.n_total_joints))

        # Joint limits from model
        self.joint_limits_lower = np.array([self.model.jnt_range[i, 0] for i in range(self.n_arm_joints)])
        self.joint_limits_upper = np.array([self.model.jnt_range[i, 1] for i in range(self.n_arm_joints)])

    def sync_joint_positions(self, joint_positions: np.ndarray):
        """Sync MuJoCo model state with real robot joint positions.

        Must be called before computing IK or FK.

        Args:
            joint_positions: Current joint positions from real robot (5 or 6 values).
        """
        # Set arm joint positions
        n_joints = min(len(joint_positions), self.n_arm_joints)
        self.data.qpos[:n_joints] = joint_positions[:n_joints]

        # Forward kinematics to update site positions
        mujoco.mj_forward(self.model, self.data)

    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position from model."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_orientation(self) -> np.ndarray:
        """Get current end-effector orientation as 3x3 rotation matrix."""
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def get_ee_euler(self) -> np.ndarray:
        """Get current end-effector orientation as euler angles (roll, pitch, yaw)."""
        R = self.get_ee_orientation()
        return self._rotation_matrix_to_euler(R)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.array([roll, pitch, yaw])

    def compute_ik(
        self,
        target_pos: np.ndarray,
        current_joints: np.ndarray,
        locked_joints: list[int] | None = None,
    ) -> np.ndarray:
        """Compute target joint positions for end-effector to reach target.

        Args:
            target_pos: Target end-effector position (3,).
            current_joints: Current joint positions (5,).
            locked_joints: List of joint indices (0-4) to keep fixed.

        Returns:
            Target joint positions (5,).
        """
        # Sync model with current joints
        self.sync_joint_positions(current_joints)

        # Position error
        current_pos = self.get_ee_position()
        pos_error = target_pos - current_pos

        # Compute Jacobian
        mujoco.mj_jacSite(
            self.model, self.data, self.jacp, self.jacr, self.ee_site_id
        )

        # Active joints
        if locked_joints is None:
            active_joints = list(range(self.n_arm_joints))
        else:
            active_joints = [i for i in range(self.n_arm_joints) if i not in locked_joints]

        n_active = len(active_joints)
        Jp = self.jacp[:, active_joints]

        # Damped least-squares
        JTJ = Jp.T @ Jp
        damping_matrix = self.damping ** 2 * np.eye(n_active)

        try:
            dq_active = np.linalg.solve(JTJ + damping_matrix, Jp.T @ pos_error)
        except np.linalg.LinAlgError:
            dq_active = np.linalg.pinv(Jp) @ pos_error

        # Clamp velocity
        dq_active = np.clip(dq_active, -self.max_dq, self.max_dq)

        # Build target joint positions
        target_joints = current_joints.copy()
        for i, joint_idx in enumerate(active_joints):
            target_joints[joint_idx] += dq_active[i]

        # Clamp to joint limits
        target_joints = np.clip(target_joints, self.joint_limits_lower, self.joint_limits_upper)

        return target_joints

    def cartesian_to_joints(
        self,
        delta_xyz: np.ndarray,
        current_joints: np.ndarray,
        action_scale: float = 0.02,
        locked_joints: list[int] | None = None,
    ) -> np.ndarray:
        """Convert Cartesian delta to target joint positions.

        This is the main interface for RL policy deployment:
        - Policy outputs delta XYZ in [-1, 1]
        - Scale by action_scale (default 2cm per unit)
        - Use IK to compute joint targets

        Args:
            delta_xyz: Policy action for XYZ (3,) in [-1, 1].
            current_joints: Current joint positions (5,).
            action_scale: Scale factor (meters per action unit).
            locked_joints: Joints to keep fixed during IK.

        Returns:
            Target joint positions (5,).
        """
        # Sync and get current EE position
        self.sync_joint_positions(current_joints)
        current_pos = self.get_ee_position()

        # Compute target position
        target_pos = current_pos + delta_xyz * action_scale

        # IK to get joint targets
        return self.compute_ik(target_pos, current_joints, locked_joints)


def test_ik():
    """Test IK controller."""
    print("Testing IK Controller...")

    try:
        ik = IKController()
        print(f"Loaded model with {ik.n_arm_joints} arm joints")

        # Test with zero joints
        joints = np.zeros(5)
        ik.sync_joint_positions(joints)

        ee_pos = ik.get_ee_position()
        ee_euler = ik.get_ee_euler()
        print(f"Zero pose - EE position: {ee_pos}")
        print(f"Zero pose - EE euler: {ee_euler}")

        # Test IK
        target = ee_pos + np.array([0.05, 0.0, 0.05])  # Move 5cm forward and up
        new_joints = ik.compute_ik(target, joints)
        print(f"IK target: {target}")
        print(f"IK result joints: {new_joints}")

        # Verify
        ik.sync_joint_positions(new_joints)
        new_pos = ik.get_ee_position()
        error = np.linalg.norm(target - new_pos)
        print(f"Position after IK: {new_pos}")
        print(f"Error: {error:.4f}m")

        print("IK Controller test passed!")

    except Exception as e:
        print(f"IK Controller test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ik()
