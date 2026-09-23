"""
End-to-end pipeline.

-> probe_video
-> load model
-> calibrate_teams 
-> process_video 
    (per frame: decode -> preprocess -> detect -> assign teams -> pressing -> render -> encode)
-> write_reports
"""

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from team_pressing.config import Config
from team_pressing.detection import Detector
from team_pressing.logging_utils import StageTimer, environment_info, seed_everything, setup_logging
from team_pressing.preprocess import preprocess
from team_pressing.pressing import RollingPressing, analyze_pressing
from team_pressing.teams import BALL, GOALKEEPER, OTHER, REFEREE, TEAM_A, TEAM_B, TeamClassifier, jersey_colors
from team_pressing.video import VideoMetadata, iter_sampled_frames, open_writer, probe_video, read_frames_at
from team_pressing.visualize import draw_frame, plot_latency, plot_pressing_timeline, save_image

# Module logger; its messages go to the handlers installed by setup_logging.
log = logging.getLogger(__name__)

# Stages timed separately inside the main loop, in execution order (log lines, metrics, chart).
STAGES = ("decode", "preprocess", "inference", "analysis", "render", "write")


def run_pipeline(video_path: str | Path, cfg: Config) -> dict:
    """
    Run every step on one video and return the run summary (also written to metrics.json).

    Args:
        video_path: input video, e.g. cut.mp4.
        cfg: validated configuration (config.yaml plus CLI overrides).
    """
    t_start = time.perf_counter()  # start of the total execution time

    # Step 1: create the run folder (never overwritten: one folder per run) and set up logging.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.output.root_dir) / run_id
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "pipeline.log", cfg.logging.level)
    seed_everything(cfg.seed)
    # Ultralytics switches OpenCV to single-threaded mode on import (a safeguard for training
    # DataLoader workers). There are no workers here, and single-threaded INTER_AREA resizing
    # costs ~19 ms/frame instead of ~4 ms on an M1, so restore OpenCV's default thread pool.
    cv2.setNumThreads(-1)

    # Step 2: record what produced this run (environment, config) for traceability.
    env = environment_info()
    log.info("Run %s | config fingerprint %s | outputs -> %s", run_id, cfg.fingerprint(), run_dir)
    log.info("Environment: %s", env)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    # Step 3: read the video metadata and plan the work.
    meta = probe_video(video_path)
    duration = min(meta.duration_s, cfg.video.max_duration_s or math.inf)  # seconds to process
    expected = math.ceil(duration * cfg.video.target_fps)  # expected number of inference frames
    log_video_metadata(meta, cfg, duration, expected)

    # Step 4: load the detector (downloads the official weights on first use).
    detector = Detector(cfg.model)

    # Step 5: learn the kit colours (the "fit" step of the team classifier).
    t0 = time.perf_counter()
    # Kit colours do not depend on the analysed range: always calibrate over the whole video.
    teams = calibrate_teams(video_path, detector, cfg, meta.duration_s)
    calibration_s = time.perf_counter() - t0

    # Step 6: main loop at 10 FPS (writes output.mp4 and the sample images).
    loop = process_video(video_path, cfg, detector, teams, run_dir, expected)
    total_s = time.perf_counter() - t_start

    # Step 7: reports (metrics.json, charts) and the final timing summary.
    summary = write_reports(run_dir, cfg, meta, detector, teams, env, loop, calibration_s, total_s)
    log.info(
        "Total execution time: %.1f s (calibration %.1f s, main loop %.1f s, %.2f inference frames/s, %.2fx real time)",
        total_s,
        calibration_s,
        loop["loop_s"],
        summary["timing"]["inference_fps"],
        summary["timing"]["realtime_factor"],
    )
    log.info("Outputs: %s", run_dir.resolve())
    return summary


def log_video_metadata(meta: VideoMetadata, cfg: Config, duration: float, expected: int) -> None:
    """
    Log the input video's properties and the processing plan (requirement: video metadata).

    Args:
        meta: probed video metadata.
        cfg: configuration (target size and FPS).
        duration: seconds of video that will be processed.
        expected: expected number of inference frames.
    """
    log.info(
        "Video: %s | %.1f MB | codec %s | %dx%d @ %.3f FPS | %d frames | %.1f s | %.2f Mbit/s",
        meta.path,
        meta.size_mb,
        meta.codec,
        meta.width,
        meta.height,
        meta.fps,
        meta.frame_count,
        meta.duration_s,
        meta.bitrate_mbps,
    )
    # Resizing to 1280x720 is required; if the source is not 16:9 the image gets stretched.
    if abs(meta.width / meta.height - cfg.video.target_width / cfg.video.target_height) > 0.01:
        log.warning("Source aspect ratio differs from target: frames will be stretched")
    log.info(
        "Plan: process %.1f s at %.1f FPS -> ~%d inference frames (1 every %.2f source frames), "
        "resized %dx%d -> %dx%d RGB",
        duration,
        cfg.video.target_fps,
        expected,
        meta.fps / cfg.video.target_fps,
        meta.width,
        meta.height,
        cfg.video.target_width,
        cfg.video.target_height,
    )


def calibrate_teams(video_path: str | Path, detector: Detector, cfg: Config, duration_s: float) -> TeamClassifier:
    """
    Learn the kit colours from people detected in frames spread over the whole video.

    Sampling the whole match (not just the first seconds) covers both halves of the pitch,
    lighting changes and moments where the referee is visible. It reuses the same
    decode -> preprocess -> detect functions as the main loop.

    Args:
        video_path: input video.
        detector: loaded detector.
        cfg: configuration (teams.* settings, target frame size, seed).
        duration_s: length of the video in seconds.
    """
    n = cfg.teams.calibration_frames
    # Centre of each of n equal segments of the video, e.g. 2.5 s, 7.5 s, ... for 60 frames of 300 s.
    timestamps = [(i + 0.5) * duration_s / n for i in range(n)]
    features, used = [], 0  # colour features per usable frame, and how many frames were usable
    for _, image in read_frames_at(video_path, timestamps):
        rgb = preprocess(image, cfg.video.target_width, cfg.video.target_height)
        dets = detector.detect(rgb)
        people = dets.select(np.isin(dets.role, ["person", "player"]))  # only people get a team
        if len(people) < cfg.teams.min_people_per_frame:
            continue  # close-ups / replays: few people, unrepresentative colours
        feats, _ = jersey_colors(rgb, people.xyxy)
        features.append(feats)
        used += 1
    if not features:
        raise RuntimeError("Team calibration found no wide shot; lower teams.min_people_per_frame")
    features = np.concatenate(features)  # (number of people, 3) Lab colours
    teams = TeamClassifier.fit(features, cfg.teams.n_clusters, cfg.teams.outlier_distance, cfg.seed)
    log.info(
        "Team calibration: %d/%d frames usable (>= %d people), %d people clustered",
        used,
        n,
        cfg.teams.min_people_per_frame,
        len(features),
    )
    for label, info in teams.summary().items():
        if isinstance(info, dict):  # skip the scalar "outlier_distance" entry
            log.info("  %-8s kit colour %s (%d samples)", label, info["hex"], info["calibration_samples"])
    return teams


def process_video(
    video_path: str | Path, cfg: Config, detector: Detector, teams: TeamClassifier, run_dir: Path, expected: int
) -> dict:
    """
    Main loop at 10 FPS: decode, preprocess, detect, assign teams, analyse, render, encode.

    Writes output.mp4 and the sample images, and logs the execution time every
    `log_every_n_frames` frames.

    Returns:
        dict with "records" (one dict per inference frame), "windows" (one timing entry per
        logging window), "stage_totals" (seconds per stage) and "loop_s" (loop wall time).
    """
    # Settings used in the loop.
    W, H, fps = cfg.video.target_width, cfg.video.target_height, cfg.video.target_fps
    window_s = cfg.pressing.smoothing_window_s  # rolling window of the pressing indicator, seconds
    rolling = RollingPressing(window=max(1, round(window_s * fps)))  # window in frames (3 s -> 30)
    every = cfg.output.log_every_n_frames  # log a timing line every N inference frames
    n_samples = cfg.output.num_sample_images  # how many annotated frames to save as images
    duration = expected / fps  # processed duration, used to spread the sample images
    saved_buckets: set[int] = set()  # time buckets that already have a sample image

    # State accumulated over the loop.
    timer = StageTimer()  # seconds spent in each stage
    records: list[dict] = []  # per-frame results (-> charts, metrics)
    windows: list[dict] = []  # per-window timings (-> latency chart)
    writer = open_writer(run_dir / cfg.output.video_name, fps, W, H, cfg.output.fourcc)
    t_loop = t_window = time.perf_counter()  # loop start, and start of the current logging window
    snapshot = dict(timer.totals)  # stage totals at the start of the current window

    def log_window(n_frames: int, timestamp_s: float) -> None:
        """Log and store the timing of the last `n_frames` frames (requirement: every 10 frames)."""
        nonlocal t_window, snapshot
        now = time.perf_counter()
        dt = now - t_window  # wall time of this window
        # Mean ms per frame for each stage during this window.
        per_stage = {s: 1000 * (timer.totals[s] - snapshot.get(s, 0.0)) / n_frames for s in STAGES}
        recent = records[-n_frames:]  # records of this window
        done = len(records)  # frames processed so far
        eta = (now - t_loop) / done * max(expected - done, 0)  # remaining time at the current pace
        log.info(
            "[frame %4d/%d | video %6.1fs] last %d frames: %.2fs (%.1f FPS) | ball %d/%d | "
            "ms/frame %s | elapsed %.0fs, ETA %.0fs",
            done,
            expected,
            timestamp_s,
            n_frames,
            dt,
            n_frames / dt,
            sum(r["ball_detected"] for r in recent),
            n_frames,
            " ".join(f"{s}={v:.1f}" for s, v in per_stage.items()),
            now - t_loop,
            eta,
        )
        windows.append(
            {
                "frame": done,
                "timestamp_s": timestamp_s,
                "seconds": dt,
                "ms_per_frame": 1000 * dt / n_frames,
                **{f"{s}_ms": v for s, v in per_stage.items()},
            }
        )
        t_window, snapshot = now, dict(timer.totals)  # start the next window

    try:
        # The iterator adds its own time to timer.totals["decode"].
        for frame in iter_sampled_frames(video_path, fps, cfg.video.max_duration_s, timings=timer.totals):
            # Step 1: resize to 720x1280 and convert/validate RGB.
            with timer("preprocess"):
                rgb = preprocess(frame.image, W, H)
            if frame.index == 0:  # log the colour/size check once
                log.info(
                    "Frame check: decoder output %s %s (BGR) -> %s %s RGB",
                    frame.image.shape,
                    frame.image.dtype,
                    rgb.shape,
                    rgb.dtype,
                )
            # Step 2: detect people and the ball.
            with timer("inference"):
                dets = detector.detect(rgb)
            # Step 3: teams, pressing triangles and the rolling indicator.
            with timer("analysis"):
                labels = teams.assign(rgb, dets)
                result = analyze_pressing(dets, labels, cfg.pressing.possession_radius)
                share = rolling.update(result.pressing_team)
            # Step 4: draw the overlay.
            with timer("render"):
                annotated = draw_frame(rgb, dets, labels, result, share, frame.timestamp_s, window_s)
            # Step 5: encode the frame into output.mp4.
            with timer("write"):
                writer.write(np.ascontiguousarray(annotated[..., ::-1]))  # RGB -> BGR for OpenCV

            # Step 6: keep a record of this frame for the reports.
            records.append(make_record(frame.index, frame.source_index, frame.timestamp_s, labels, dets, result, share))

            # Step 7: a few illustrative frames: the first frame with both triangles in each time bucket.
            bucket = min(int(frame.timestamp_s / duration * n_samples), n_samples - 1) if n_samples else -1
            both = all(t is not None for t in result.triangles.values())
            if n_samples and both and bucket not in saved_buckets:
                saved_buckets.add(bucket)
                m, s = divmod(int(frame.timestamp_s), 60)
                save_image(run_dir / "visualizations" / f"sample_{bucket + 1}_{m:02d}m{s:02d}s.png", annotated)

            # Step 8: timing log every N frames.
            if len(records) % every == 0:
                log_window(every, frame.timestamp_s)
    finally:
        # Always close the writer, even on error or Ctrl+C, so output.mp4 stays playable.
        writer.release()

    # Log the last, partial window (when the frame count is not a multiple of N).
    if records and len(records) % every:
        log_window(len(records) % every, records[-1]["timestamp_s"])
    return {
        "records": records,
        "windows": windows,
        "stage_totals": dict(timer.totals),
        "loop_s": time.perf_counter() - t_loop,
    }


def make_record(index, source_index, timestamp_s, labels, dets, result, share) -> dict:
    """
    Flatten one frame's results into a dict (input of the charts and of the metrics).

    Args:
        index: position in the 10 FPS sequence.
        source_index: position in the original video.
        timestamp_s: time in the original video.
        labels: final label per detection.
        dets: detections of the frame.
        result: pressing analysis of the frame.
        share: rolling share of verdicts, or None.
    """

    def count(label):
        """Number of detections with this label."""
        return int(np.sum(labels == label))

    tri_a, tri_b = result.triangles[TEAM_A], result.triangles[TEAM_B]
    ball_conf = dets.conf[labels == BALL]  # confidence of the kept ball (0 or 1 value)
    return {
        "frame": index,
        "source_frame": source_index,
        "timestamp_s": round(timestamp_s, 3),
        "n_team_a": count(TEAM_A),
        "n_team_b": count(TEAM_B),
        "n_referee": count(REFEREE),
        "n_goalkeeper": count(GOALKEEPER),
        "n_other": count(OTHER),
        "ball_detected": result.ball is not None,
        "ball_x": None if result.ball is None else round(float(result.ball[0]), 1),
        "ball_y": None if result.ball is None else round(float(result.ball[1]), 1),
        "ball_conf": round(float(ball_conf[0]), 3) if len(ball_conf) else None,
        "possessor_team": result.possessor_team,
        "area_a": None if tri_a is None else round(tri_a.area, 1),
        "area_b": None if tri_b is None else round(tri_b.area, 1),
        "pressing_team": result.pressing_team,
        "rolling_share_a": None if share is None else round(share[TEAM_A], 3),
    }


def write_reports(
    run_dir: Path,
    cfg: Config,
    meta: VideoMetadata,
    detector: Detector,
    teams: TeamClassifier,
    env: dict,
    loop: dict,
    calibration_s: float,
    total_s: float,
) -> dict:
    """
    Write the charts and metrics.json; log the analysis summary; return the summary dict.

    Args:
        run_dir: run folder.
        cfg: configuration.
        meta: video metadata.
        detector: detector (for model info: weights, hash, device).
        teams: fitted team classifier (for the learned kit colours).
        env: environment info (versions, git commit).
        loop: output of process_video.
        calibration_s: calibration duration in seconds.
        total_s: total execution time in seconds.
    """
    records, windows = loop["records"], loop["windows"]
    n = len(records)  # number of inference frames
    vis = run_dir / "visualizations"

    # Charts.
    ms_per_frame = {s: 1000 * loop["stage_totals"].get(s, 0.0) / max(n, 1) for s in STAGES}  # mean ms per stage
    window = max(1, round(cfg.pressing.smoothing_window_s * cfg.video.target_fps))  # rolling window, frames
    if records:
        plot_pressing_timeline(records, window, vis / "pressing_timeline.png")
        plot_latency(windows, ms_per_frame, 1000 / cfg.video.target_fps, vis / "latency.png")

    # Aggregates for the summary.
    decided = [r["pressing_team"] for r in records if r["pressing_team"]]  # verdict per decided frame
    areas = {t: [r[k] for r in records if r[k] is not None] for t, k in ((TEAM_A, "area_a"), (TEAM_B, "area_b"))}
    video_s = records[-1]["timestamp_s"] + 1 / cfg.video.target_fps if records else 0.0  # seconds covered
    summary = {
        "run_id": run_dir.name,
        "config_fingerprint": cfg.fingerprint(),
        "environment": env,
        "video": meta.as_dict(),
        "model": detector.info(),
        "teams": teams.summary(),
        "processing": {
            "inference_frames": n,
            "video_seconds_processed": round(video_s, 2),
            "target_fps": cfg.video.target_fps,
            "frame_size": [cfg.video.target_height, cfg.video.target_width],
        },
        "timing": {
            "total_s": round(total_s, 2),
            "calibration_s": round(calibration_s, 2),
            "main_loop_s": round(loop["loop_s"], 2),
            "inference_fps": round(n / loop["loop_s"], 2) if loop["loop_s"] else 0.0,
            # > 1 means faster than real time (video seconds processed per wall-clock second).
            "realtime_factor": round(video_s / loop["loop_s"], 3) if loop["loop_s"] else 0.0,
            "ms_per_frame": {s: round(v, 2) for s, v in ms_per_frame.items()},
        },
        "analysis": {
            "frames_with_ball": sum(r["ball_detected"] for r in records),
            "frames_with_triangle": {TEAM_A: len(areas[TEAM_A]), TEAM_B: len(areas[TEAM_B])},
            "frames_decided": len(decided),
            "pressing_share": {
                t: round(decided.count(t) / len(decided), 3) if decided else None for t in (TEAM_A, TEAM_B)
            },
            "median_triangle_area_px2": {t: round(float(np.median(a)), 1) if a else None for t, a in areas.items()},
            "possession_frames": {t: sum(r["possessor_team"] == t for r in records) for t in (TEAM_A, TEAM_B)},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Human-readable summary in the log.
    a = summary["analysis"]
    log.info(
        "Analysis: ball in %d/%d frames, pressing decided in %d frames -> Team A %s, Team B %s",
        a["frames_with_ball"],
        n,
        a["frames_decided"],
        *(f"{v:.0%}" if v is not None else "n/a" for v in a["pressing_share"].values()),
    )
    log.info("Mean ms/frame by stage: %s", ", ".join(f"{s}={v:.1f}" for s, v in ms_per_frame.items()))
    return summary
