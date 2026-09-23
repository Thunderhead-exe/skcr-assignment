"""
Typed and validated configuration loaded from config.yaml.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Roles a detector class can be mapped to (model.class_map):
# - person: a COCO person; the jersey colour decides team_a / team_b / referee / other
# - player / goalkeeper / referee: classes of a football fine-tuned detector
# - ball: the ball
Role = Literal["person", "player", "goalkeeper", "referee", "ball"]


class _Strict(BaseModel):
    """Base class of every section: unknown keys are errors, so a typo in config.yaml fails loudly."""

    model_config = ConfigDict(extra="forbid")


class VideoConfig(_Strict):
    """Input handling: output frame size, sampling rate and optional time limit."""

    target_width: int = Field(1280, gt=0)
    target_height: int = Field(720, gt=0)
    target_fps: float = Field(10.0, gt=0)
    # Option to process only the first N seconds (None = the whole video)
    max_duration_s: float | None = Field(None, gt=0)


class ModelConfig(_Strict):
    """Detector: weights, device, input size, class mapping and confidence thresholds."""

    # Path to the YOLO weights
    weights: str = "models/yolo26s.pt"
    # "auto" or an explicit device such as "cpu", "mps", "cuda:0"
    device: str = "auto"
    # Network input size in pixels
    imgsz: int = Field(1280, gt=0)
    # Model class id -> pipeline role. Classes not listed are ignored by the detector.
    class_map: dict[int, Role] = {0: "person", 32: "ball"}
    # Minimum confidence per role (a detection below its role's threshold is dropped).
    conf: dict[Role, float] = {"person": 0.3, "ball": 0.15}

    @model_validator(mode="after")
    def _check_roles(self) -> "ModelConfig":
        """Cross-field checks: the analysis needs a ball class, and every mapped role needs a threshold."""
        if "ball" not in self.class_map.values():
            raise ValueError("model.class_map must map one class to 'ball'")
        missing = set(self.class_map.values()) - set(self.conf)  # roles without a threshold
        if missing:
            raise ValueError(f"model.conf is missing thresholds for roles: {sorted(missing)}")
        return self


class TeamsConfig(_Strict):
    """Team calibration by jersey-colour clustering."""

    # Number of frames sampled evenly across the video to learn the kit colours.
    calibration_frames: int = Field(60, ge=5)
    # A calibration frame is used only if it has at least this many people (wide shots, not close-ups).
    min_people_per_frame: int = Field(8, ge=1)
    # K-means clusters: 3 = two teams + referees (COCO model), 2 = two teams (fine-tuned model).
    n_clusters: int = Field(3, ge=2, le=3)
    # Lab colour distance above which a person matches no cluster and is labelled "other".
    outlier_distance: float = Field(40.0, gt=0)


class PressingConfig(_Strict):
    """Pressing analysis."""

    # A player has the ball when it is within this fraction of their box height from their feet.
    possession_radius: float = Field(0.5, gt=0)
    # Length in seconds of the rolling window used to smooth the per-frame verdicts.
    smoothing_window_s: float = Field(3.0, gt=0)


class OutputConfig(_Strict):
    """Where and how results are written."""

    # Parent folder of the run folders (each run writes to <root_dir>/<run_id>/).
    root_dir: str = "outputs"
    # File name of the annotated video inside the run folder.
    video_name: str = "output.mp4"
    # Video codecs (FourCC codes) tried in order; the first one this OpenCV build supports is used.
    fourcc: list[str] = ["avc1", "mp4v"]
    # Log the execution time every N inference frames (requirement: 10).
    log_every_n_frames: int = Field(10, ge=1)
    # Number of annotated frames saved as PNG in visualizations/ (0 = none).
    num_sample_images: int = Field(6, ge=0)


class LoggingConfig(_Strict):
    """Logging verbosity."""

    # Minimum level written to the console and to pipeline.log.
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Config(_Strict):
    """The whole configuration: one attribute per section of config.yaml."""

    # Random seed (k-means initialisation), for reproducible team clustering.
    seed: int = 42
    video: VideoConfig = VideoConfig()
    model: ModelConfig = ModelConfig()
    teams: TeamsConfig = TeamsConfig()
    pressing: PressingConfig = PressingConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()

    def fingerprint(self) -> str:
        """
        Short hash of the resolved config, logged with every run for traceability.
        Two runs with the same fingerprint used exactly the same settings.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)  # key order does not matter
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    """
    Read and validate a YAML config file (an empty file gives the defaults).
    Raises:
        pydantic.ValidationError: on unknown keys, wrong types or out-of-range values.
    """
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f) or {})
