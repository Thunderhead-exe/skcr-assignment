"""
Frame preprocessing: resize to the target resolution and guarantee 3-channel RGB uint8.
"""

import cv2
import numpy as np

# (channel order of the input, number of channels) -> OpenCV conversion code to RGB.
# 3-channel RGB input needs no conversion, so it is not listed.
_TO_RGB = {
    ("BGR", 3): cv2.COLOR_BGR2RGB,
    ("BGR", 4): cv2.COLOR_BGRA2RGB,
    ("RGB", 4): cv2.COLOR_RGBA2RGB,
}


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize to exactly (height, width).

    INTER_AREA when shrinking: it averages source pixels, avoiding aliasing. 
    Bilinear when enlarging. Returns the input unchanged if already the right size.
    """
    if frame.shape[:2] == (height, width):
        return frame
    shrinking = frame.shape[0] > height or frame.shape[1] > width
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interp)  # OpenCV takes (width, height)


def ensure_rgb(frame: np.ndarray, source_order: str = "BGR") -> np.ndarray:
    """
    Return a contiguous HxWx3 uint8 RGB image, whatever the decoder produced.

    Handles grayscale (HxW or HxWx1), BGR/RGB and BGRA/RGBA inputs, and non-uint8 depth.

    Args:
        frame: decoded image.
        source_order: channel order of colour input, "BGR" (OpenCV) or "RGB".

    Raises:
        ValueError: if the shape is not an image shape.
    """
    # Step 1: bit depth. E.g. 10-bit video decoded to uint16: scale to 8 bits.
    if frame.dtype != np.uint8:
        max_val = np.iinfo(frame.dtype).max if np.issubdtype(frame.dtype, np.integer) else 1.0  # floats: 0..1
        frame = np.clip(frame.astype(np.float32) * (255.0 / max_val), 0, 255).astype(np.uint8)

    # Step 2: channels.
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):  # grayscale: replicate to 3 channels
        rgb = cv2.cvtColor(frame.reshape(frame.shape[:2]), cv2.COLOR_GRAY2RGB)
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):  # colour, with or without alpha
        code = _TO_RGB.get((source_order, frame.shape[2]))
        rgb = cv2.cvtColor(frame, code) if code is not None else frame  # None: already RGB
    else:
        raise ValueError(f"Unsupported frame shape {frame.shape}")

    # Step 3: check the result, and make it contiguous (reversed-channel views are not).
    validate_rgb(rgb)
    return np.ascontiguousarray(rgb)


def validate_rgb(frame: np.ndarray, height: int | None = None, width: int | None = None) -> None:
    """
    Contract check between pipeline steps: raises instead of passing on a bad frame.

    Args:
        frame: image to check.
        height, width: expected size; the size is only checked when both are given.

    Raises:
        ValueError: if the frame is not HxWx3 uint8, or not of the expected size.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 RGB frame, got shape={frame.shape} dtype={frame.dtype}")
    if height is not None and width is not None and frame.shape[:2] != (height, width):
        raise ValueError(f"Expected {height}x{width} frame, got {frame.shape[0]}x{frame.shape[1]}")


def preprocess(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Decoder output (BGR) -> resized, validated RGB frame.

    Resizing first makes the colour conversion cheaper (0.9 MP instead of 2 MP for 1080p input).
    """
    rgb = ensure_rgb(resize_frame(frame_bgr, width, height), source_order="BGR")
    validate_rgb(rgb, height, width)  # guarantees the 720x1280 requirement to the next steps
    return rgb
