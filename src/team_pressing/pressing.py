"""
Pressing analysis: per team, the closest triangle of players that encloses the ball.

For each team we look at every triangle formed by three of its players (their foot
points, i.e. where they stand on the pitch), keep those that contain the ball, and pick
the "closest" one: the smallest sum of distances from the ball to the three players.
Its area measures how tightly that team surrounds the ball: when both teams have such a
triangle, the team with the smaller one is the one pressing more in that frame.
"""

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from team_pressing.detection import Detections
from team_pressing.teams import BALL, TEAM_A, TEAM_B, TEAMS


@dataclass
class Triangle:
    """
    One team's closest triangle enclosing the ball.

    Attributes:
        vertices: (3, 2) foot points of the three players, in pixels.
        members: (3,) indices of these players in the frame's detections.
        area: triangle area in px².
    """

    vertices: np.ndarray
    members: np.ndarray
    area: float


@dataclass
class PressingResult:
    """
    Pressing analysis of one frame

    Attributes:
        ball: (2,) ball position in pixels, or None if no ball was detected.
        possessor: detection index of the ball carrier, or None if nobody has the ball.
        possessor_team: team of the ball carrier, or None.
        triangles: closest enclosing triangle per team (None if that team does not enclose the ball).
        pressing_team: team pressing more in this frame, or None when there is no verdict.
    """

    ball: np.ndarray | None = None
    possessor: int | None = None
    possessor_team: str | None = None
    triangles: dict[str, Triangle | None] = field(default_factory=lambda: {TEAM_A: None, TEAM_B: None})
    pressing_team: str | None = None


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """
    Bottom-centre of each box: the player's position on the ground plane

    Args:
        xyxy: (N, 4) boxes.

    Returns:
        (N, 2) points (x, y) in pixels.
    """
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)


def triangle_areas(tris: np.ndarray) -> np.ndarray:
    """Areas of (M, 3, 2) triangles with the shoelace formula: |cross(b - a, c - a)| / 2."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]  # the three vertices of every triangle
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def triangles_contain(tris: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    (M,) bool: point p inside (or on the edge of) each triangle.

    For each edge, the sign of a cross product tells on which side of the edge p lies.
    p is inside when it is on the same side of all three edges (no mix of signs).
    """

    def cross(o, q):
        """Cross product of (q - o) and (p - o) for every triangle: > 0 left of the edge o->q, < 0 right."""
        return (q[:, 0] - o[:, 0]) * (p[1] - o[:, 1]) - (q[:, 1] - o[:, 1]) * (p[0] - o[:, 0])

    d1, d2, d3 = cross(tris[:, 0], tris[:, 1]), cross(tris[:, 1], tris[:, 2]), cross(tris[:, 2], tris[:, 0])
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)  # p is right of at least one edge
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)  # p is left of at least one edge
    return ~(has_neg & has_pos)


def closest_enclosing_triangle(points: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    Among all triangles of `points` that contain `target`, the one whose vertices are closest to it.

    N <= ~11 players, so testing all C(N, 3) <= 165 triangles vectorised is instant.

    Args:
        points: (N, 2) player positions of one team.
        target: (2,) ball position.

    Returns:
        (indices of its 3 points, area), or None if no triangle contains the target.
    """
    if len(points) < 3:
        return None  # a triangle needs three players
    combos = np.array(list(combinations(range(len(points)), 3)))  # (M, 3) index triples
    tris = points[combos]  # (M, 3, 2) vertices of every candidate triangle
    areas = triangle_areas(tris)
    valid = triangles_contain(tris, target) & (areas > 1.0)  # contains the ball, not degenerate (collinear)
    if not valid.any():
        return None
    cost = np.linalg.norm(tris - target, axis=2).sum(axis=1)  # sum of ball-to-vertex distances
    cost[~valid] = np.inf  # never pick a triangle that does not contain the ball
    best = int(np.argmin(cost))
    return combos[best], float(areas[best])


def find_possessor(feet: np.ndarray, heights: np.ndarray, ball: np.ndarray, radius: float) -> int | None:
    """
    Index of the player whose feet are closest to the ball, if within radius x their height.

    The threshold scales with the player's box height, so it adapts to the distance from the
    camera (players far away are smaller), unlike a fixed pixel radius.

    Args:
        feet: (N, 2) foot points.
        heights: (N,) box heights in pixels.
        ball: (2,) ball position.
        radius: fraction of the box height (config: pressing.possession_radius).
    """
    if len(feet) == 0:
        return None
    dist = np.linalg.norm(feet - ball, axis=1)  # distance from each player's feet to the ball
    close = dist <= radius * heights  # players close enough to have the ball
    if not close.any():
        return None
    return int(np.flatnonzero(close)[np.argmin(dist[close])])  # the closest of them


def analyze_pressing(dets: Detections, labels: np.ndarray, possession_radius: float) -> PressingResult:
    """
    Full pressing analysis of one frame: ball, carrier, one triangle per team, verdict.

    Args:
        dets: detections of the frame.
        labels: final label per detection (from TeamClassifier.assign).
        possession_radius: see find_possessor.
    """
    result = PressingResult()

    # Step 1: the ball. Without it there is nothing to analyse.
    ball_idx = np.flatnonzero(labels == BALL)
    if len(ball_idx) == 0:
        return result
    b = dets.xyxy[ball_idx[0]]  # ball box
    result.ball = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)  # centre of the box

    # Step 2: the ball carrier, among team players only (not referees or others).
    players = np.flatnonzero(np.isin(labels, TEAMS))  # detection indices of team players
    feet = foot_points(dets.xyxy[players])
    heights = dets.xyxy[players, 3] - dets.xyxy[players, 1]
    p = find_possessor(feet, heights, result.ball, possession_radius)  # index into `players`
    if p is not None:
        result.possessor = int(players[p])
        result.possessor_team = str(labels[result.possessor])

    # Step 3: each team's closest triangle enclosing the ball.
    for team in TEAMS:
        # The ball carrier cannot be a vertex of a triangle that must contain the ball:
        # the carrier's team triangle is formed by the supporting teammates around them.
        members = [i for i in players if labels[i] == team and i != result.possessor]
        found = closest_enclosing_triangle(foot_points(dets.xyxy[members]), result.ball) if members else None
        if found is not None:
            idx, area = found  # indices into `members`, and area
            chosen = np.array(members)[idx]  # back to detection indices
            result.triangles[team] = Triangle(foot_points(dets.xyxy[chosen]), chosen, area)

    # Step 4: verdict. Compare only when both teams enclose the ball. If only one does, it is usually
    # the team in possession surrounding its own ball carrier (support, not pressing), so no verdict.
    tri_a, tri_b = result.triangles[TEAM_A], result.triangles[TEAM_B]
    if tri_a is not None and tri_b is not None:
        result.pressing_team = TEAM_A if tri_a.area < tri_b.area else TEAM_B
    return result


class RollingPressing:
    """
    Share of the decided frames in the last `window` frames won by each team.

    Per-frame decisions are noisy (a missed detection flips a triangle), so the overlay
    and the timeline show this rolling share instead.
    """

    def __init__(self, window: int):
        """Args: window: number of frames in the rolling window (3 s x 10 FPS = 30)."""
        # Verdict of each recent frame (None = no verdict); the oldest drops out automatically.
        self.history: deque[str | None] = deque(maxlen=window)

    def update(self, pressing_team: str | None) -> dict[str, float] | None:
        """Add this frame's verdict and return {team: share} over the window, or None if no verdict yet."""
        self.history.append(pressing_team)
        decided = [t for t in self.history if t is not None]  # frames of the window with a verdict
        if not decided:
            return None
        return {team: sum(t == team for t in decided) / len(decided) for team in TEAMS}
