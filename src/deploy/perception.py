"""Perception module for segmentation and depth estimation.

Provides SegmentationModel, DepthModel, and SegDepthPreprocessor classes
for real-time inference with seg+depth observations.
"""

import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PICK101_ROOT = Path("/home/gota/ggando/ml/pick-101")
DEPTH_ANYTHING_DIR = PROJECT_ROOT / "third_party" / "Depth-Anything-V2"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


class SegmentationModel:
    """Wrapper for EfficientViT segmentation model.

    Loads model from pick-101 checkpoint and runs inference on BGR images.

    Class mapping (from real segmentation model):
        0: unlabeled
        1: background
        2: table
        3: cube
        4: static_finger
        5: moving_finger
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        debug: bool = False,
    ):
        """Initialize segmentation model.

        Args:
            checkpoint_path: Path to EfficientViT checkpoint (.ckpt file).
            device: Torch device for inference.
            debug: Enable debug logging.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.debug = debug
        self._model = None
        self._call_count = 0

    def load(self) -> bool:
        """Load model from checkpoint.

        Returns:
            True if loaded successfully.
        """
        if not self.checkpoint_path.exists():
            print(f"Segmentation checkpoint not found: {self.checkpoint_path}")
            return False

        if self.debug:
            print(f"[SEG DEBUG] Loading from: {self.checkpoint_path}")
            print(f"[SEG DEBUG] Device: {self.device}")
            print(f"[SEG DEBUG] sys.path (first 5):")
            for i, p in enumerate(sys.path[:5]):
                print(f"  [{i}] {p}")

        # Add pick-101 paths (both root and training dir for relative imports)
        pick101_str = str(PICK101_ROOT)
        training_str = str(PICK101_ROOT / "src" / "training")

        # Remove if present, then insert at front
        for p in [pick101_str, training_str]:
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, training_str)
        sys.path.insert(0, pick101_str)

        if self.debug:
            print(f"[SEG DEBUG] sys.path after manipulation (first 5):")
            for i, p in enumerate(sys.path[:5]):
                print(f"  [{i}] {p}")

        try:
            from infer_efficientvit_seg import SegmentationInference

            if self.debug:
                import infer_efficientvit_seg
                print(f"[SEG DEBUG] SegmentationInference from: {infer_efficientvit_seg.__file__}")

            self._model = SegmentationInference(
                str(self.checkpoint_path), device=self.device
            )

            if self.debug:
                print(f"[SEG DEBUG] Model img_height: {self._model.img_height}")
                print(f"[SEG DEBUG] Model img_width: {self._model.img_width}")
                print(f"[SEG DEBUG] Model num_classes: {self._model.num_classes}")
                print(f"[SEG DEBUG] Model device: {self._model.device}")

            print(f"Loaded segmentation model: {self.checkpoint_path.name}")
            return True
        except Exception as e:
            print(f"Failed to load segmentation model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def predict(self, bgr_image: np.ndarray) -> np.ndarray:
        """Run segmentation inference.

        Args:
            bgr_image: BGR image (H, W, 3) from OpenCV.

        Returns:
            Segmentation mask (H, W) with class IDs 0-5.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        self._call_count += 1

        if self.debug and self._call_count <= 3:
            print(f"[SEG DEBUG] predict() call #{self._call_count}")
            print(f"  Input shape: {bgr_image.shape}, dtype: {bgr_image.dtype}")
            print(f"  Input range: [{bgr_image.min()}, {bgr_image.max()}]")
            print(f"  Input mean (BGR): [{bgr_image[:,:,0].mean():.1f}, {bgr_image[:,:,1].mean():.1f}, {bgr_image[:,:,2].mean():.1f}]")

        mask = self._model.predict(bgr_image)

        if self.debug and self._call_count <= 3:
            unique, counts = np.unique(mask, return_counts=True)
            print(f"  Output shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"  Output classes: {dict(zip(unique, counts))}")

        return mask


class DepthModel:
    """Wrapper for Depth Anything V2 Small model.

    Produces disparity (inverse depth) matching training format:
    - Output normalized per-frame to [0, 255]
    - Near objects = 255, far objects = 0
    """

    CHECKPOINT_PATH = CHECKPOINT_DIR / "depth_anything_v2_vits.pth"

    def __init__(self, device: str = "cuda"):
        """Initialize depth model.

        Args:
            device: Torch device for inference.
        """
        self.device = device
        self._model = None

    def load(self) -> bool:
        """Load Depth Anything V2 Small model.

        Returns:
            True if loaded successfully.
        """
        if not DEPTH_ANYTHING_DIR.exists():
            print(f"Depth Anything V2 not found at {DEPTH_ANYTHING_DIR}")
            print("Run: uv run python scripts/depth_estimation.py --setup")
            return False

        if not self.CHECKPOINT_PATH.exists():
            print(f"Depth checkpoint not found at {self.CHECKPOINT_PATH}")
            print("Run: uv run python scripts/depth_estimation.py --setup")
            return False

        try:
            if str(DEPTH_ANYTHING_DIR) not in sys.path:
                sys.path.insert(0, str(DEPTH_ANYTHING_DIR))

            # Suppress xFormers warnings (printed to stderr during import)
            import io
            import contextlib
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                from depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
            }

            self._model = DepthAnythingV2(**model_configs["vits"])
            self._model.load_state_dict(
                torch.load(str(self.CHECKPOINT_PATH), map_location="cpu")
            )
            self._model = self._model.to(self.device).eval()

            print(f"Loaded depth model: {self.CHECKPOINT_PATH.name}")
            return True
        except Exception as e:
            print(f"Failed to load depth model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def predict(self, bgr_image: np.ndarray) -> np.ndarray:
        """Run depth estimation and convert to disparity.

        Args:
            bgr_image: BGR image (H, W, 3) from OpenCV.

        Returns:
            Disparity map (H, W) uint8, values 0-255 (near=255, far=0).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            # Model expects BGR input, outputs depth/disparity
            depth = self._model.infer_image(bgr_image)

        # Normalize per-frame to 0-255 (same as training)
        # Depth Anything outputs relative disparity where higher = closer
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            disparity_norm = (depth - d_min) / (d_max - d_min)
        else:
            disparity_norm = np.ones_like(depth)

        disparity_uint8 = (disparity_norm * 255).astype(np.uint8)
        return disparity_uint8


class SegDepthPreprocessor:
    """Preprocesses real camera images to seg+depth observations.

    Pipeline (matching training exactly):
    1. Capture BGR frame (640x480)
    2. Center crop to square (480x480)
    3. Run segmentation -> (480, 480) class IDs
    4. Run depth estimation -> (480, 480) disparity
    5. Resize to 84x84
    6. Stack as 2-channel: (2, 84, 84)
    7. Frame stacking: (frame_stack, 2, 84, 84)
    """

    def __init__(
        self,
        seg_checkpoint: str | Path,
        target_size: tuple[int, int] = (84, 84),
        frame_stack: int = 3,
        camera_index: int = 0,
        device: str = "cuda",
    ):
        """Initialize preprocessor.

        Args:
            seg_checkpoint: Path to segmentation model checkpoint.
            target_size: Output image size (H, W).
            frame_stack: Number of frames to stack.
            camera_index: OpenCV camera index.
            device: Torch device for inference.
        """
        self.target_size = target_size
        self.frame_stack = frame_stack
        self.camera_index = camera_index
        self.device = device

        # Models
        self._seg_model = SegmentationModel(seg_checkpoint, device=device)
        self._depth_model = DepthModel(device=device)

        # Camera
        self._cap: Optional[cv2.VideoCapture] = None

        # Frame buffer for stacking
        self._frame_buffer: deque = deque(maxlen=frame_stack)

        # Last raw frame for recording
        self._last_raw_frame: Optional[np.ndarray] = None

        # Last seg/depth for preview visualization
        self._last_seg_mask: Optional[np.ndarray] = None
        self._last_disparity: Optional[np.ndarray] = None

    def load_models(self) -> bool:
        """Load segmentation and depth models.

        Returns:
            True if both models loaded successfully.
        """
        if not self._seg_model.load():
            return False
        if not self._depth_model.load():
            return False
        return True

    def open_camera(self) -> bool:
        """Open camera capture.

        Returns:
            True if camera opened successfully.
        """
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def close(self):
        """Release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def warm_up(self, num_frames: int = 10):
        """Warm up camera and models by running inference on initial frames."""
        print("  Warming up perception models...")
        for i in range(num_frames):
            if self._cap is not None:
                ret, frame = self._cap.read()
                if ret:
                    # Center crop before warm-up (same as inference)
                    cropped = self._center_crop_square(frame)
                    _ = self._seg_model.predict(cropped)
                    _ = self._depth_model.predict(cropped)
        print("  Warm up complete.")

    def reset(self):
        """Clear frame buffer."""
        self._frame_buffer.clear()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Get last captured raw BGR frame."""
        return self._last_raw_frame

    def get_preview_image(self, preview_size: int = 240) -> Optional[np.ndarray]:
        """Get preview image showing RGB | Seg | Depth side by side.

        Args:
            preview_size: Size of each panel (square).

        Returns:
            BGR image (preview_size, preview_size*3, 3) or None.
        """
        if self._last_raw_frame is None:
            return None

        # Segmentation class colors (BGR for OpenCV)
        seg_colors = np.array(
            [
                [0, 0, 0],  # 0: unlabeled - black
                [0, 255, 0],  # 1: background - green
                [255, 0, 255],  # 2: table - magenta
                [255, 127, 0],  # 3: cube - orange (BGR)
                [0, 127, 255],  # 4: static_finger - cyan (BGR)
                [127, 255, 127],  # 5: moving_finger - light green
            ],
            dtype=np.uint8,
        )

        # RGB panel (center crop and resize)
        rgb_cropped = self._center_crop_square(self._last_raw_frame)
        rgb_panel = cv2.resize(rgb_cropped, (preview_size, preview_size))

        # Seg panel (colorize)
        if self._last_seg_mask is not None:
            seg_colored = seg_colors[self._last_seg_mask]
            seg_panel = cv2.resize(seg_colored, (preview_size, preview_size))

            # Add color legend
            class_names = ["unlbl", "bg", "table", "cube", "s_fgr", "m_fgr"]
            legend_y = preview_size - 12  # Bottom of panel
            legend_x = 5
            font_small = cv2.FONT_HERSHEY_SIMPLEX
            for i, (name, color) in enumerate(zip(class_names, seg_colors)):
                # Draw colored square
                x = legend_x + i * 40
                cv2.rectangle(seg_panel, (x, legend_y - 8), (x + 8, legend_y), tuple(int(c) for c in color), -1)
                cv2.rectangle(seg_panel, (x, legend_y - 8), (x + 8, legend_y), (255, 255, 255), 1)
                # Draw label
                cv2.putText(seg_panel, name, (x + 10, legend_y), font_small, 0.3, (255, 255, 255), 1)
        else:
            seg_panel = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)

        # Depth panel (colormap)
        if self._last_disparity is not None:
            depth_colored = cv2.applyColorMap(self._last_disparity, cv2.COLORMAP_INFERNO)
            depth_panel = cv2.resize(depth_colored, (preview_size, preview_size))
        else:
            depth_panel = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)

        # Combine: RGB | Seg | Depth
        combined = np.hstack([rgb_panel, seg_panel, depth_panel])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "RGB", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(
            combined, "Seg", (preview_size + 10, 25), font, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            combined, "Depth", (preview_size * 2 + 10, 25), font, 0.7, (255, 255, 255), 2
        )

        return combined

    def _center_crop_square(self, image: np.ndarray) -> np.ndarray:
        """Center crop to square (takes center 480x480 from 640x480)."""
        h, w = image.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return image[y_start : y_start + size, x_start : x_start + size]

    def _preprocess_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame to seg+depth observation.

        Args:
            bgr_frame: BGR frame from camera (480, 640, 3).

        Returns:
            Observation (2, 84, 84) uint8.
        """
        # Center crop to square FIRST (480x480 from 640x480)
        # This ensures segmentation runs on same aspect ratio as training
        bgr_cropped = self._center_crop_square(bgr_frame)

        # Run segmentation on cropped square -> (480, 480) class IDs 0-5
        seg_mask = self._seg_model.predict(bgr_cropped)

        # Run depth on cropped square -> (480, 480) disparity 0-255
        disparity = self._depth_model.predict(bgr_cropped)

        # Store for preview (before resize)
        self._last_seg_mask = seg_mask
        self._last_disparity = disparity

        # Resize to target size
        # INTER_NEAREST for segmentation (preserve class IDs)
        seg_mask_resized = cv2.resize(
            seg_mask,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        # INTER_LINEAR for depth (smooth gradients)
        disparity_resized = cv2.resize(
            disparity,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Stack channels: (2, H, W)
        obs = np.stack([seg_mask_resized, disparity_resized], axis=0)
        return obs.astype(np.uint8)

    def capture_and_preprocess(self, max_retries: int = 5) -> Optional[np.ndarray]:
        """Capture frame and preprocess to observation.

        Args:
            max_retries: Number of capture retries on failure.

        Returns:
            Observation (2, 84, 84) uint8 or None if capture failed.
        """
        if self._cap is None:
            return None

        for attempt in range(max_retries):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._last_raw_frame = frame.copy()
                return self._preprocess_frame(frame)
            time.sleep(0.05)  # Brief delay before retry

        return None

    def _flush_camera_buffer(self, num_frames: int = 5):
        """Discard stale frames from camera buffer."""
        if self._cap is None:
            return
        for _ in range(num_frames):
            self._cap.read()

    def fill_buffer(self) -> bool:
        """Fill frame buffer with initial frames.

        Returns:
            True if buffer filled successfully.
        """
        # Flush stale frames first
        self._flush_camera_buffer()
        time.sleep(0.1)  # Let camera stabilize

        for i in range(self.frame_stack):
            frame = self.capture_and_preprocess(max_retries=10)
            if frame is None:
                print(f"    [Perception] Failed to capture frame {i+1}/{self.frame_stack}")
                return False
            self._frame_buffer.append(frame)
        return True

    def get_stacked_observation(self) -> Optional[np.ndarray]:
        """Get frame-stacked observation for policy.

        Returns:
            Stacked frames (3, 2, 84, 84) uint8 via np.stack.
        """
        frame = self.capture_and_preprocess()
        if frame is None:
            return None

        self._frame_buffer.append(frame)

        if len(self._frame_buffer) < self.frame_stack:
            return None

        # Stack frames: 3 frames of (2, 84, 84) -> (3, 2, 84, 84)
        # Agent's flatten_time_dim_into_channel_dim will flatten internally
        stacked = np.stack(list(self._frame_buffer), axis=0)
        return stacked


class MockSegDepthPreprocessor:
    """Mock preprocessor for dry-run testing without camera/models."""

    def __init__(
        self,
        target_size: tuple[int, int] = (84, 84),
        frame_stack: int = 3,
    ):
        self.target_size = target_size
        self.frame_stack = frame_stack
        self._frame_buffer: deque = deque(maxlen=frame_stack)

    def load_models(self) -> bool:
        print("  [MOCK] Using mock perception models")
        return True

    def open_camera(self) -> bool:
        print("  [MOCK] Using mock camera")
        return True

    def close(self):
        pass

    def warm_up(self, num_frames: int = 10):
        pass

    def reset(self):
        self._frame_buffer.clear()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def get_preview_image(self, preview_size: int = 240) -> Optional[np.ndarray]:
        """Generate mock preview showing random RGB | Seg | Depth."""
        # Generate mock panels
        rgb_panel = np.random.randint(0, 256, (preview_size, preview_size, 3), dtype=np.uint8)

        # Mock segmentation with random class colors
        seg_colors = np.array([
            [0, 0, 0], [0, 255, 0], [255, 0, 255],
            [255, 127, 0], [0, 127, 255], [127, 255, 127]
        ], dtype=np.uint8)
        seg_ids = np.random.randint(0, 6, (preview_size, preview_size), dtype=np.uint8)
        seg_panel = seg_colors[seg_ids]

        # Mock depth colormap
        depth = np.random.randint(0, 256, (preview_size, preview_size), dtype=np.uint8)
        depth_panel = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        # Combine
        combined = np.hstack([rgb_panel, seg_panel, depth_panel])

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "RGB [MOCK]", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Seg [MOCK]", (preview_size + 10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Depth [MOCK]", (preview_size * 2 + 10, 25), font, 0.6, (255, 255, 255), 2)

        return combined

    def fill_buffer(self) -> bool:
        for _ in range(self.frame_stack):
            frame = self._generate_mock_frame()
            self._frame_buffer.append(frame)
        return True

    def _generate_mock_frame(self) -> np.ndarray:
        """Generate random seg+depth observation."""
        seg = np.random.randint(0, 6, self.target_size, dtype=np.uint8)
        depth = np.random.randint(0, 256, self.target_size, dtype=np.uint8)
        return np.stack([seg, depth], axis=0)

    def get_stacked_observation(self) -> Optional[np.ndarray]:
        frame = self._generate_mock_frame()
        self._frame_buffer.append(frame)

        if len(self._frame_buffer) < self.frame_stack:
            return None

        # Stack frames: 3 frames of (2, 84, 84) -> (3, 2, 84, 84)
        stacked = np.stack(list(self._frame_buffer), axis=0)
        return stacked
