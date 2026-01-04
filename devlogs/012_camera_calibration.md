# Wrist Camera Calibration for Sim-to-Real

## Camera Hardware

**innoMaker 1080P USB2.0 UVC Camera**
- Resolution: 1920x1080 @ 5fps, 640x480 @ 30fps
- FOV: 130° diagonal, 103° horizontal
- Interface: USB 2.0, UVC standard
- Size: 32x32mm

## FOV Calibration

### The Problem

MuJoCo was showing more of the gripper body than the real camera despite matching position and angle. The real camera only showed gripper fingers.

### Root Cause

MuJoCo's `cam_fovy` is **vertical** FOV, not horizontal.

The innoMaker specs list:
- 130° diagonal FOV
- 103° horizontal FOV

For 4:3 aspect ratio (640x480), converting horizontal to vertical:
```
vertical_fov = 2 * atan(tan(horizontal_fov/2) / aspect_ratio)
vertical_fov = 2 * atan(tan(51.5°) / 1.333)
vertical_fov ≈ 86°
```

### Fix

```python
model.cam_fovy[cam_id] = 86.0  # NOT 103.0
```

## Camera Pitch Calculation

Camera geometry on SO-101:
- Camera offset: 7cm behind gripper center
- Look-at point: 13cm forward of gripper center
- Total horizontal distance: 20cm

Pitch angle:
```
pitch = -atan(vertical_offset / horizontal_distance)
pitch = -atan(h / 20cm)
```

With measured vertical offset ~9.8cm:
```
pitch ≈ -26°
```

## Image Preprocessing

### Pipeline

```
Real camera (640x480, 4:3)
    → Center crop to square (480x480)
    → Resize to 84x84
    → BGR to RGB
    → HWC to CHW format
    → Frame stack (3 frames)
```

### Center Crop

The real camera is 4:3, but MuJoCo renders 1:1. Center crop removes horizontal edges:

```python
def center_crop_square(image):
    h, w = image.shape[:2]
    size = min(h, w)  # 480 for 640x480
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return image[y_start:y_start+size, x_start:x_start+size]
```

## Final Camera Settings (MuJoCo)

```python
# Position: floating behind gripper, matching real camera mount
cam_pos = [x, y, z]  # Match physical mount position

# Orientation
cam_fovy = 86.0      # Vertical FOV (NOT horizontal 103°)
cam_pitch = -26.0    # Looking down at workspace

# Render
render_width = 84
render_height = 84
```

## Visual Alignment Check

After calibration, both sim and real should show:
- Both gripper fingers visible
- Static finger base near bottom-right corner
- Workspace/objects visible in center-forward area

## Files

- `src/deploy/camera.py` - CameraPreprocessor with center crop + resize
- `scripts/record_camera.py` - Debug tool for capturing real camera frames
