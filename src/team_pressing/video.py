"""
Video I/O: metadata probing, 10 FPS frame sampling, decoding and encoding.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# Module logger.
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """
    Properties of the input video (logged, and saved in metrics.json).

    Attributes:
        path: video path as given on the command line.
        size_mb: file size in megabytes.
        codec: FourCC codec name, e.g. "h264".
        width, height: frame size in pixels.
        fps: frames per second declared by the container.
        frame_count: number of frames declared by the container.
        duration_s: frame_count / fps.
        bitrate_mbps: average bitrate, from file size and duration.
    """

    path: str
    size_mb: float
    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    bitrate_mbps: float

    def as_dict(self) -> dict:
        """Plain dict version (for JSON)."""
        return asdict(self)


@dataclass(frozen=True)
class SampledFrame:
    """
    One frame selected for inference.

    Attributes:
        index: position in the sampled (10 FPS) sequence: 0, 1, 2, ...
        source_index: position in the original video (e.g. 0, 3, 5, 8, ... at 25 FPS).
        timestamp_s: presentation time in the original video, in seconds.
        image: decoded frame, as returned by OpenCV (BGR).
    """

    index: int
    source_index: int
    timestamp_s: float
    image: np.ndarray


def probe_video(path: str | Path) -> VideoMetadata:
    """
    Read the video's properties without decoding it.

    Raises:
        FileNotFoundError / RuntimeError: if the file is missing or cannot be opened.
    """
    cap = _open_capture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # codec as a 32-bit integer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    size_mb = Path(path).stat().st_size / 1e6
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        path=str(path),
        size_mb=round(size_mb, 2),
        codec=fourcc.to_bytes(4, "little").decode(errors="replace").strip("\x00"),  # int -> "h264"
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_s=round(duration, 3),
        bitrate_mbps=round(size_mb * 8 / duration, 2) if duration else 0.0,  # MB -> Mbit, per second
    )


class FrameSampler:
    """
    Selects frames so that inference runs at `target_fps` per second of *video time*.

    It works on timestamps rather than a fixed frame stride: 25 FPS -> 10 FPS is a
    2.5 stride, so "every Nth frame" would give 12.5 or 8.3 inferences per second.
    Taking the first frame at or after each target instant k / target_fps gives exactly
    10 per second for any source rate (25, 29.97, 50, variable frame rate...).
    """

    def __init__(self, target_fps: float):
        """Args: target_fps: inferences per second of video (10)."""
        self.period = 1.0 / target_fps  # seconds between target instants (0.1 s)
        self._k = 0  # index of the next target instant (its time is k * period)

    def accept(self, timestamp_s: float) -> bool:
        """True if the frame at `timestamp_s` should be inferred. Call it for every frame, in order."""
        # 1e-6 s tolerance absorbs floating-point error (e.g. 5 * 0.04 vs 0.2).
        if timestamp_s + 1e-6 < self._k * self.period:
            return False  # next target instant not reached yet
        # Skip past every target instant this frame covers (matters only if target > source fps
        # or the stream has gaps), so one frame is never used twice.
        while self._k * self.period <= timestamp_s + 1e-6:
            self._k += 1
        return True


def iter_sampled_frames(
    path: str | Path, target_fps: float, max_duration_s: float | None = None, timings: dict | None = None
) -> Iterator[SampledFrame]:
    """
    Decode the video sequentially and yield only the frames selected by FrameSampler.

    Skipped frames are only `grab()`bed (demuxed and decoded, not converted into numpy
    arrays), which is the cheap path in OpenCV.

    Args:
        path: input video.
        target_fps: inferences per second of video.
        max_duration_s: stop after this many seconds of video (None = whole video).
        timings: if given, decode time is added to timings["decode"].
    """
    cap = _open_capture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # used only if the container has no timestamps
    sampler = FrameSampler(target_fps)
    source_index, sample_index = -1, 0  # counters: frames read, frames yielded
    try:
        while True:
            t0 = time.perf_counter()
            # Step 1: read the next frame (compressed data decoded, but not converted to an array yet).
            if not cap.grab():
                break  # end of the video
            source_index += 1
            # Step 2: its timestamp from the container (handles variable frame rate); fall back to index / fps.
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 or source_index / fps
            if max_duration_s is not None and timestamp >= max_duration_s:
                break
            # Step 3: convert to a numpy array only if the frame is selected.
            selected = sampler.accept(timestamp)
            if selected:
                ok, image = cap.retrieve()
            if timings is not None:
                timings["decode"] = timings.get("decode", 0.0) + time.perf_counter() - t0
            if not selected:
                continue
            if not ok or image is None:
                log.warning("Could not decode frame %d, skipping it", source_index)
                continue
            # Step 4: hand the frame to the caller; decoding resumes when the caller asks for the next one.
            yield SampledFrame(sample_index, source_index, timestamp, image)
            sample_index += 1
    finally:
        cap.release()  # also runs if the caller stops early


def read_frames_at(path: str | Path, timestamps_s: list[float]) -> Iterator[tuple[float, np.ndarray]]:
    """
    Random access by seeking; used by team calibration to sample the whole match.

    Yields (timestamp, BGR frame) for each timestamp that could be decoded.
    """
    cap = _open_capture(path)
    try:
        for t in timestamps_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)  # seek (OpenCV decodes from the previous keyframe)
            ok, image = cap.read()
            if ok and image is not None:
                yield t, image
    finally:
        cap.release()


def open_writer(path: str | Path, fps: float, width: int, height: int, fourccs: list[str]) -> cv2.VideoWriter:
    """
    Open an MP4 writer with the first codec available in this OpenCV build.

    H.264 ('avc1') plays everywhere but is missing from most Linux OpenCV wheels;
    'mp4v' (MPEG-4 Part 2) is always available.

    Args:
        path: output file (output.mp4).
        fps: frame rate of the output video (10, so it lasts as long as the source).
        width, height: frame size.
        fourccs: codecs to try, in order.

    Raises:
        RuntimeError: if no codec works.
    """
    for fourcc in fourccs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if writer.isOpened():
            log.info("Video writer: codec=%s fps=%.1f size=%dx%d -> %s", fourcc, fps, width, height, path)
            return writer
        writer.release()
        log.warning("Codec %s unavailable in this OpenCV build, trying next", fourcc)
    raise RuntimeError(f"No usable codec among {fourccs} for {path}")


def _open_capture(path: str | Path) -> cv2.VideoCapture:
    """Open a video for reading, with clear errors for a missing or unreadable file."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    return cap
