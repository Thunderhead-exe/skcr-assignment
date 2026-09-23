"""
Rendering: annotated frames for output.mp4 and summary charts (PNG).

Two kinds of output are produced here:
- per-frame overlays (boxes, pressing triangles, ball marker, info panel) drawn with OpenCV
  on the RGB frame, which the pipeline then encodes into output.mp4;
- end-of-run charts (pressing timeline, latency) drawn with matplotlib.
"""

from pathlib import Path

import cv2
import numpy as np  # noqa: E402  (array maths for the overlays and the charts)
import matplotlib
import matplotlib.pyplot as plt  

from team_pressing.detection import Detections
from team_pressing.pressing import PressingResult
from team_pressing.teams import BALL, GOALKEEPER, OTHER, REFEREE, TEAM_A, TEAM_B

matplotlib.use("Agg") # "Agg" renders to files without a display, so the charts also work on servers, in Docker and in CI.

### colours and labels -----------------------------------------------------------------------------

# One colour per label. Teams (and goalkeepers, which only a fine-tuned model detects) use a
# categorical palette checked for colour-vision deficiency; officials and others are neutral,
# so on the overlay colour always means "team".
HEX = {TEAM_A: "#eb6834", TEAM_B: "#2a78d6", GOALKEEPER: "#4a3aa7", REFEREE: "#111111", OTHER: "#d0d0d0"}
# The same colours as (R, G, B) tuples for OpenCV (frames are RGB inside the pipeline).
RGB = {k: tuple(int(v[i : i + 2], 16) for i in (1, 3, 5)) for k, v in HEX.items()}
# Human-readable names used in the info panel and the charts.
NAMES = {TEAM_A: "Team A", TEAM_B: "Team B", REFEREE: "Referee", GOALKEEPER: "Goalkeeper", OTHER: "Other"}
# Short names used in the compact legend of the info panel.
SHORT = {TEAM_A: "A", TEAM_B: "B", GOALKEEPER: "GK", REFEREE: "Ref", OTHER: "Other"}
# Chart text colours (primary ink, secondary ink) and gridline colour.
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# OpenCV font used for all overlay text.
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Ball marker: a disc slightly larger than a ball in a wide shot (~8-10 px at 720p), blended at
# low opacity so the real ball stays visible underneath it.
BALL_MARKER_RADIUS = 9
BALL_MARKER_OPACITY = 0.35


def save_image(path: Path, frame_rgb: np.ndarray) -> None:
    """Write an RGB frame to disk (PNG/JPG from the extension). OpenCV expects BGR, hence the conversion."""
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


### frame overlay -----------------------------------------------------------------------------


def draw_frame(
    frame_rgb: np.ndarray,
    dets: Detections,
    labels: np.ndarray,
    result: PressingResult,
    rolling_share: dict[str, float] | None,
    timestamp_s: float,
    window_s: float,
) -> np.ndarray:
    """
    Return an annotated copy of the frame (the input frame is not modified).

    Args:
        frame_rgb: 720x1280 RGB frame.
        dets: detections of this frame.
        labels: final label per detection (team_a, team_b, referee, other, goalkeeper, ball).
        result: pressing analysis of this frame (ball, carrier, triangles, verdict).
        rolling_share: share of recent verdicts won by each team, or None if no verdict yet.
        timestamp_s: position of the frame in the source video, shown in the panel.
        window_s: length of the rolling window, shown in the panel.
    """
    out = frame_rgb.copy()  # work on a copy so the caller's frame stays clean

    # Step 1: pressing triangles, drawn first so boxes and the ball stay on top of them.
    triangles = [(team, tri) for team, tri in result.triangles.items() if tri is not None]
    if triangles:
        overlay = out.copy()  # fill on a copy, then blend: translucent fill
        for team, tri in triangles:
            cv2.fillPoly(overlay, [tri.vertices.round().astype(np.int32)], RGB[team])
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)  # 30 % fill opacity
        for team, tri in triangles:  # solid outline on top of the translucent fill
            cv2.polylines(out, [tri.vertices.round().astype(np.int32)], True, RGB[team], 2, cv2.LINE_AA)

    # Step 2: one box per person, coloured by label; the ball carrier gets a thick box and a marker.
    for i, (box, label) in enumerate(zip(dets.xyxy.round().astype(int), labels, strict=True)):
        if label == BALL:
            continue  # the ball gets its own marker below
        x1, y1, x2, y2 = box  # box corners in pixels
        color = RGB.get(str(label), RGB[OTHER])  # unknown labels fall back to the neutral colour
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if i == result.possessor else 1)
        if i == result.possessor:  # small downward triangle above the ball carrier's head
            cx = (x1 + x2) // 2  # horizontal centre of the box
            pts = np.array([[cx - 7, y1 - 14], [cx + 7, y1 - 14], [cx, y1 - 4]], np.int32)
            cv2.fillPoly(out, [pts], color)

    # Step 3: translucent ball marker, so the detection is visible without hiding the real ball.
    if result.ball is not None:
        _draw_ball_marker(out, result.ball)

    # Step 4: info panel (time, areas, verdict, rolling share, legend).
    _draw_hud(out, result, rolling_share, timestamp_s, window_s, show_gk=bool(np.any(labels == GOALKEEPER)))
    return out


def _draw_ball_marker(img: np.ndarray, ball_xy: np.ndarray) -> None:
    """
    Draw a translucent white disc with a dark outline centred on the ball, in place.

    Only a small patch around the ball is blended (not the whole frame), which keeps it cheap.
    """
    cx, cy = int(ball_xy[0]), int(ball_xy[1])  # ball centre in pixels
    r = BALL_MARKER_RADIUS
    h, w = img.shape[:2]
    # Patch around the ball, clipped to the frame (the ball can be near an edge).
    x1, y1 = max(cx - r - 2, 0), max(cy - r - 2, 0)
    x2, y2 = min(cx + r + 3, w), min(cy + r + 3, h)
    if x1 >= x2 or y1 >= y2:
        return  # ball centre outside the frame: nothing to draw
    patch = img[y1:y2, x1:x2]  # view into the frame: writing to it edits the frame
    overlay = patch.copy()  # draw the opaque marker on a copy of the patch...
    centre = (cx - x1, cy - y1)  # ball centre in patch coordinates
    cv2.circle(overlay, centre, r, (255, 255, 255), -1, cv2.LINE_AA)  # white disc
    cv2.circle(overlay, centre, r, (0, 0, 0), 1, cv2.LINE_AA)  # dark outline for contrast on white lines
    # ...then blend it back: BALL_MARKER_OPACITY of the marker, the rest of the original pixels.
    patch[:] = cv2.addWeighted(overlay, BALL_MARKER_OPACITY, patch, 1 - BALL_MARKER_OPACITY, 0)


def _draw_hud(
    img: np.ndarray, result: PressingResult, share: dict | None, timestamp_s: float, window_s: float, show_gk: bool
) -> None:
    """
    Draw the info panel in the bottom-right corner, in place.

    Rows: title and time, Team A and Team B triangle areas, this frame's verdict, rolling-share
    bar, and a legend (the goalkeeper entry only appears when goalkeepers are detected).
    """
    h, w = img.shape[:2]  # frame size
    pw, ph = 330, 152  # panel width and height in pixels
    x0, y0 = w - pw - 12, h - ph - 12  # panel top-left corner, 12 px from the frame edges
    panel = img[y0 : y0 + ph, x0 : x0 + pw]  # view into the frame
    panel[:] = (panel * 0.35).astype(np.uint8)  # darken the area: translucent dark background

    def text(s, x, y, scale=0.48):
        """White text at (x, y) relative to the panel's top-left corner."""
        cv2.putText(img, s, (x0 + x, y0 + y), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def swatch(label, x, y):
        """Small colour square for `label`, with a light border so dark colours stay visible."""
        cv2.rectangle(img, (x0 + x, y0 + y - 10), (x0 + x + 12, y0 + y + 2), RGB[label], -1)
        cv2.rectangle(img, (x0 + x, y0 + y - 10), (x0 + x + 12, y0 + y + 2), (200, 200, 200), 1)

    # Title row with the video time as mm:ss.s.
    minutes, seconds = divmod(timestamp_s, 60)
    text(f"Pressing analysis   {int(minutes):02d}:{seconds:04.1f}", 10, 20, 0.5)

    # One row per team with its triangle area ("-" when the team does not enclose the ball).
    for row, team in enumerate((TEAM_A, TEAM_B)):
        tri = result.triangles[team]
        area = f"{tri.area:,.0f} px^2" if tri is not None else "-"
        swatch(team, 10, 44 + row * 20)
        text(f"{NAMES[team]} triangle: {area}", 30, 44 + row * 20)

    # Verdict of this frame, or why there is none.
    if result.ball is None:
        status = "Ball not detected"
    elif result.pressing_team is None:
        status = "No verdict: needs both triangles"
    else:
        status = f"Pressing more: {NAMES[result.pressing_team]}"
    text(status, 10, 88)

    # Rolling share bar: Team A's share fills from the left, Team B's from the right.
    bx, by, bw, bh = 10, 100, pw - 20, 16  # bar position and size inside the panel
    if share is None:
        text(f"last {window_s:.0f}s: no verdict yet", bx, by + 12, 0.42)
    else:
        split = int(round(bw * share[TEAM_A]))  # x position where Team A's part ends
        cv2.rectangle(img, (x0 + bx, y0 + by), (x0 + bx + split, y0 + by + bh), RGB[TEAM_A], -1)
        cv2.rectangle(img, (x0 + bx + split, y0 + by), (x0 + bx + bw, y0 + by + bh), RGB[TEAM_B], -1)
        text(f"last {window_s:.0f}s  A {share[TEAM_A]:.0%} | B {share[TEAM_B]:.0%}", bx + 6, by + 12, 0.42)

    # Legend row: one swatch + short name per label, then the ball marker.
    x = 10  # running x position of the next legend entry
    for label in (TEAM_A, TEAM_B, GOALKEEPER, REFEREE, OTHER):
        if label == GOALKEEPER and not show_gk:
            continue  # COCO models never output goalkeepers: do not advertise the colour
        swatch(label, x, 140)
        text(SHORT[label], x + 17, 140, 0.42)
        x += 24 + 9 * len(SHORT[label])  # advance by swatch + text width
    cv2.circle(img, (x0 + x + 6, y0 + 136), 6, (255, 255, 255), -1, cv2.LINE_AA)
    text("Ball", x + 17, 140, 0.42)


### charts -----------------------------------------------------------------------------


def _style(ax) -> None:
    """Shared chart styling: no top/right frame, light solid gridlines behind the data, muted ticks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)


def _rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    """
    Trailing rolling median over `window` samples that ignores NaN (frames without a triangle).

    Returns NaN where the whole window is NaN, so the line has gaps where there is no data.
    """
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]  # the last `window` values up to i
        if np.isfinite(chunk).any():
            out[i] = np.nanmedian(chunk)
    return out


def plot_pressing_timeline(records: list[dict], window: int, path: Path) -> None:
    """
    Two-panel chart of the pressing analysis over the video, saved as PNG.

    Top: each team's enclosing-triangle area per frame (dots) and its rolling median (line),
    on a log scale because areas range from a few thousand to a few hundred thousand px².
    Bottom: rolling share of verdicts. Above 50 % (orange) Team A is tighter around the ball,
    below 50 % (blue) Team B is.

    Args:
        records: per-frame records from the pipeline (see pipeline.make_record).
        window: rolling window length in frames (3 s x 10 FPS = 30).
        path: output PNG path.
    """
    t = np.array([r["timestamp_s"] for r in records]) / 60.0  # x axis in minutes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6.5), height_ratios=[1.2, 1])

    # Top panel: triangle areas.
    for team, key in ((TEAM_A, "area_a"), (TEAM_B, "area_b")):
        area = np.array([np.nan if r[key] is None else r[key] for r in records], float)  # NaN = no triangle
        ax1.scatter(t, area, s=6, color=HEX[team], alpha=0.2, linewidths=0)
        ax1.plot(t, _rolling_nanmedian(area, window), color=HEX[team], lw=2, label=f"{NAMES[team]}")
    if any(r["area_a"] or r["area_b"] for r in records):
        ax1.set_yscale("log")
    else:  # e.g. a short clip of close-ups: keep a valid (linear) axis and say why it is empty
        ax1.text(0.5, 0.5, "No enclosing triangle in this range", transform=ax1.transAxes, ha="center", color=INK_2)
    ax1.set_ylabel("Triangle area (px², log)", color=INK_2)
    ax1.set_title(
        "Closest triangle enclosing the ball: dots = frames, line = rolling median", loc="left", color=INK, fontsize=11
    )
    ax1.legend(frameon=False, loc="upper right", fontsize=9)

    # Bottom panel: rolling share of verdicts won by Team A, filled towards the winning side.
    share = np.array([np.nan if r["rolling_share_a"] is None else r["rolling_share_a"] for r in records])
    ax2.fill_between(t, 0.5, share, where=share >= 0.5, color=HEX[TEAM_A], alpha=0.35, lw=0, interpolate=True)
    ax2.fill_between(t, 0.5, share, where=share < 0.5, color=HEX[TEAM_B], alpha=0.35, lw=0, interpolate=True)
    ax2.plot(t, share, color=INK_2, lw=1.2)
    ax2.axhline(0.5, color=INK_2, lw=0.8)  # 50/50 reference line
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1], ["100% B", "75% B", "50/50", "75% A", "100% A"])
    ax2.set_title("Who presses more (rolling share of analysed frames)", loc="left", color=INK, fontsize=11)
    ax2.text(0.005, 0.93, "▲ Team A tighter around the ball", transform=ax2.transAxes, color=INK_2, fontsize=9)
    ax2.text(0.005, 0.03, "▼ Team B tighter around the ball", transform=ax2.transAxes, color=INK_2, fontsize=9)
    ax2.set_xlabel("Video time (min)", color=INK_2)

    # Headline: overall share of verdicts for the whole run.
    decided = [r["pressing_team"] for r in records if r["pressing_team"]]  # frames with a verdict
    if decided:
        pa = sum(d == TEAM_A for d in decided) / len(decided)  # Team A's share of the verdicts
        fig.suptitle(
            f"Team A pressed more in {pa:.0%} of analysed frames, Team B in {1 - pa:.0%} "
            f"({len(decided)} of {len(records)} frames analysed)",
            x=0.01,
            ha="left",
            fontsize=13,
            color=INK,
        )
    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)  # free the figure's memory


def plot_latency(windows: list[dict], ms_per_frame: dict[str, float], budget_ms: float, path: Path) -> None:
    """
    Latency chart saved as PNG: ms/frame over the video (left) and mean ms per stage (right).

    Args:
        windows: one entry per logging window, with "timestamp_s" and "ms_per_frame".
        ms_per_frame: mean milliseconds per inference frame for each pipeline stage.
        budget_ms: real-time budget per frame (1000 / target_fps), drawn as a reference line.
        path: output PNG path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2, 1])

    # Left panel: end-to-end ms/frame for each 10-frame window, against the real-time budget.
    t = [w["timestamp_s"] / 60 for w in windows]  # x axis in minutes
    ax1.plot(t, [w["ms_per_frame"] for w in windows], color=HEX[TEAM_B], lw=2)
    ax1.axhline(budget_ms, color=INK_2, lw=1)
    ax1.text(
        t[-1] if t else 0,
        budget_ms * 1.02,
        f"real-time budget at 10 FPS: {budget_ms:.0f} ms/frame",
        color=INK_2,
        fontsize=9,
        va="bottom",
        ha="right",
    )
    ax1.set_ylim(0, max([budget_ms * 1.25] + [w["ms_per_frame"] * 1.1 for w in windows]))
    ax1.set_xlabel("Video time (min)", color=INK_2)
    ax1.set_ylabel("ms per inference frame", color=INK_2)
    ax1.set_title("End-to-end latency, averaged over each 10-frame window", loc="left", color=INK, fontsize=11)

    # Right panel: horizontal bars of mean ms per stage, with the value at the tip of each bar.
    stages = list(ms_per_frame)  # stage names in pipeline order
    values = [ms_per_frame[s] for s in stages]
    ax2.barh(stages, values, height=0.5, color=HEX[TEAM_B])
    ax2.invert_yaxis()  # first stage at the top
    for y, v in enumerate(values):
        ax2.text(v, y, f" {v:.1f}", va="center", color=INK, fontsize=9)
    ax2.set_xlim(0, max(values) * 1.25 if values else 1)  # room for the value labels
    ax2.set_xlabel("mean ms per frame", color=INK_2)
    ax2.set_title("Where the time goes", loc="left", color=INK, fontsize=11)

    for ax in (ax1, ax2):
        _style(ax)
    ax2.grid(False, axis="y")  # horizontal gridlines add nothing to a bar list
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)  # free the figure's memory
