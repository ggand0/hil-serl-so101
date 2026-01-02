"""Camera capture and preprocessing for sim-to-real transfer.

Handles the visual domain gap between MuJoCo rendering and real camera.
"""

from collections import deque
from typing import Optional

import cv2
import numpy as np


class CameraPreprocessor:
    """Preprocesses real camera images to match sim observations.

    Sim training uses:
    - 84x84 RGB images from wrist_cam
    - 75 degree FOV (approximately)
    - Frame stacking of 3 frames

    Real camera (innoMaker):
    - 640x480 or 1080p
    - 130 degree diagonal FOV
    - Need to crop to match sim FOV

    Pipeline:
        Real camera → center crop (FOV matching) → resize to 84x84 → CHW format → frame stack
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (84, 84),
        frame_stack: int = 3,
        sim_fov: float = 75.0,
        real_fov: float = 130.0,
        camera_index: int = 0,
    ):
        """Initialize camera preprocessor.

        Args:
            target_size: Output image size (H, W). Default (84, 84).
            frame_stack: Number of frames to stack. Default 3.
            sim_fov: Sim camera field of view in degrees. Default 75.
            real_fov: Real camera diagonal FOV in degrees. Default 130.
            camera_index: OpenCV camera index. Default 0.
        """
        self.target_size = target_size
        self.frame_stack = frame_stack
        self.sim_fov = sim_fov
        self.real_fov = real_fov
        self.camera_index = camera_index

        # Crop ratio to match FOV
        # This is approximate - diagonal FOV vs horizontal FOV differ
        self.crop_ratio = sim_fov / real_fov

        # Frame buffer for stacking
        self._frame_buffer: deque = deque(maxlen=frame_stack)

        # Camera capture
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open camera capture.

        Returns:
            True if camera opened successfully.
        """
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            return False

        # Set resolution (640x480 is common for USB cameras)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        return True

    def close(self):
        """Release camera capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """Center crop image to match sim FOV.

        Args:
            image: Input image (H, W, C).

        Returns:
            Center-cropped image.
        """
        h, w = image.shape[:2]

        # Calculate crop dimensions
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)

        # Calculate crop offsets
        y_start = (h - crop_h) // 2
        x_start = (w - crop_w) // 2

        return image[y_start : y_start + crop_h, x_start : x_start + crop_w]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame.

        Args:
            frame: BGR frame from OpenCV (H, W, 3).

        Returns:
            Preprocessed frame (3, 84, 84) uint8.
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Center crop to match sim FOV
        cropped = self._center_crop(rgb)

        # Resize to target size
        resized = cv2.resize(
            cropped, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA
        )

        # Convert to CHW format (H, W, C) -> (C, H, W)
        chw = np.transpose(resized, (2, 0, 1))

        return chw.astype(np.uint8)

    def capture_and_preprocess(self) -> Optional[np.ndarray]:
        """Capture a frame from camera and preprocess it.

        Returns:
            Preprocessed frame (3, 84, 84) or None if capture failed.
        """
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        return self._preprocess_frame(frame)

    def get_stacked_observation(self) -> Optional[np.ndarray]:
        """Get frame-stacked observation.

        Captures a new frame, adds to buffer, and returns stacked frames.

        Returns:
            Stacked frames (frame_stack, 3, H, W) or None if not enough frames.
        """
        frame = self.capture_and_preprocess()
        if frame is None:
            return None

        self._frame_buffer.append(frame)

        # Need full buffer for frame stacking
        if len(self._frame_buffer) < self.frame_stack:
            return None

        # Stack frames: (frame_stack, C, H, W)
        stacked = np.stack(list(self._frame_buffer), axis=0)
        return stacked

    def reset(self):
        """Clear frame buffer (call at episode start)."""
        self._frame_buffer.clear()

    def warm_up(self, num_frames: int = 10):
        """Capture and discard frames to warm up camera.

        Some cameras need a few frames before auto-exposure stabilizes.

        Args:
            num_frames: Number of frames to capture and discard.
        """
        for _ in range(num_frames):
            if self._cap is not None:
                self._cap.read()

    def fill_buffer(self) -> bool:
        """Fill frame buffer with initial frames.

        Call this at episode start after reset() to initialize the buffer.

        Returns:
            True if buffer filled successfully.
        """
        for _ in range(self.frame_stack):
            frame = self.capture_and_preprocess()
            if frame is None:
                return False
            self._frame_buffer.append(frame)
        return True


def test_camera():
    """Test camera capture and preprocessing."""
    preprocessor = CameraPreprocessor(camera_index=0)

    if not preprocessor.open():
        print("Failed to open camera")
        return

    print("Camera opened. Warming up...")
    preprocessor.warm_up()

    print("Filling frame buffer...")
    preprocessor.reset()
    if not preprocessor.fill_buffer():
        print("Failed to fill buffer")
        preprocessor.close()
        return

    print("Capturing stacked observation...")
    obs = preprocessor.get_stacked_observation()
    if obs is not None:
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Observation range: [{obs.min()}, {obs.max()}]")
    else:
        print("Failed to get observation")

    preprocessor.close()
    print("Camera closed.")


if __name__ == "__main__":
    test_camera()
