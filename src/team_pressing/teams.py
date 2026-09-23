"""
Unsupervised team / referee assignment from jersey colours.

COCO detectors only know "person". Each person's shirt colour is summarised as the
mean CIELAB colour of the torso (grass pixels removed). K-means over people sampled
across the match gives one cluster per kit: the two largest clusters are the teams,
the next one the referees (they appear in almost every wide shot, unlike goalkeepers).
People far from every cluster (goalkeepers, staff, ball kids) become "other".
"""

from dataclasses import dataclass

import cv2
import numpy as np

from team_pressing.detection import Detections

# Final labels a detection can have after this step (used across the whole pipeline).
TEAM_A, TEAM_B, REFEREE, OTHER, GOALKEEPER, BALL = "team_a", "team_b", "referee", "other", "goalkeeper", "ball"
# The two team labels, in a fixed order.
TEAMS = (TEAM_A, TEAM_B)

# Grass in OpenCV HSV (H in 0..180): green hue, reasonably saturated. These pixels are
# removed from the shirt crop so the pitch does not bias the jersey colour.
_GRASS_HUE = (36, 86)  # hue range of grass
_GRASS_MIN_SAT = 50  # minimum saturation: excludes white/grey shirt pixels with a green tint


def jersey_color(frame_rgb: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """
    Mean Lab colour of the shirt area (upper-middle part of the box), grass excluded.

    Args:
        frame_rgb: RGB frame.
        box: person box (x1, y1, x2, y2) in pixels.

    Returns:
        (3,) float32 Lab colour in OpenCV's 8-bit scale, or None if the crop is too small.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1  # box width and height
    # Torso: 15-50 % of the height (below the head, above the shorts), central 50 % of the width
    # (away from the arms and the background at the box edges).
    top, bottom = int(y1 + 0.15 * h), int(y1 + 0.50 * h)
    left, right = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    H, W = frame_rgb.shape[:2]  # frame size, to clip the crop inside the image
    crop = frame_rgb[max(top, 0) : min(bottom, H), max(left, 0) : min(right, W)]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None  # person too small or cut by the frame edge
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)  # one row per pixel, for the grass test
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)  # colour feature space
    grass = (hsv[:, 0] >= _GRASS_HUE[0]) & (hsv[:, 0] <= _GRASS_HUE[1]) & (hsv[:, 1] >= _GRASS_MIN_SAT)
    kept = lab[~grass]  # shirt pixels only
    # If almost everything looks like grass (e.g. a green kit), fall back to all pixels.
    return (kept if len(kept) >= 5 else lab).mean(axis=0)


def jersey_colors(frame_rgb: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Features for several boxes.

    Returns:
        (features (M, 3), indices of the boxes that yielded a feature). M can be smaller than
        the number of boxes when some crops are too small.
    """
    feats, idx = [], []
    for i, box in enumerate(boxes):
        f = jersey_color(frame_rgb, box)
        if f is not None:
            feats.append(f)
            idx.append(i)
    return np.array(feats, np.float32).reshape(-1, 3), np.array(idx, int)


def lab_to_hex(lab: np.ndarray) -> str:
    """Convert an OpenCV 8-bit Lab colour to "#rrggbb" (for logs and metrics.json)."""
    rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB)[0, 0]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@dataclass
class TeamClassifier:
    """
    Kit colours learned by k-means, and the rule that assigns a label to each person.

    Attributes:
        centroids: (k, 3) Lab colour of each cluster, rows in the same order as `labels`.
        labels: label of each cluster: [team_a, team_b] (+ [referee] when k = 3).
        sizes: number of calibration samples in each cluster (a sanity check for the log).
        outlier_distance: Lab distance beyond which a person matches no cluster -> "other".
    """

    centroids: np.ndarray
    labels: list[str]
    sizes: list[int]
    outlier_distance: float

    @classmethod
    def fit(cls, features: np.ndarray, n_clusters: int, outlier_distance: float, seed: int) -> "TeamClassifier":
        """
        Cluster the calibration colours and name the clusters.

        Args:
            features: (N, 3) Lab colours of people from the calibration frames.
            n_clusters: 3 for a COCO model (2 teams + referees), 2 for a fine-tuned model.
            outlier_distance: see the class attributes.
            seed: random seed, so k-means gives the same clusters on every run.
        """
        if len(features) < n_clusters * 5:
            raise ValueError(f"Not enough people to calibrate teams ({len(features)} samples)")
        cv2.setRNGSeed(seed)  # k-means++ initialisation is random
        # Stop after 100 iterations or when centres move less than 0.1 Lab units.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        # 10 attempts with different initialisations; the most compact clustering is kept.
        _, assign, centers = cv2.kmeans(
            features.astype(np.float32), n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        sizes = np.bincount(assign.ravel(), minlength=n_clusters)  # samples per cluster
        by_size = list(np.argsort(-sizes))  # cluster ids, largest first
        # The two largest clusters are the teams. Deterministic naming: Team A is the lighter
        # kit (higher L channel), so the same team gets the same name on every run.
        teams = sorted(by_size[:2], key=lambda c: -centers[c, 0])
        order = teams + by_size[2:]  # [team A, team B, referees]
        return cls(
            centroids=centers[order],
            labels=[TEAM_A, TEAM_B] + [REFEREE] * (n_clusters - 2),
            sizes=[int(sizes[c]) for c in order],
            outlier_distance=outlier_distance,
        )

    def assign(self, frame_rgb: np.ndarray, dets: Detections) -> np.ndarray:
        """
        Final label per detection: team_a | team_b | referee | goalkeeper | other | ball.

        Args:
            frame_rgb: RGB frame the detections come from.
            dets: detections with their detector role.
        """
        labels = np.array(dets.role, dtype="<U10")  # start from the detector's roles
        # Roles already known by the detector (ball, and with a fine-tuned model goalkeeper and
        # referee) are kept. Only "person"/"player" need a team; default them to "other".
        labels[np.isin(labels, ["person", "player"])] = OTHER
        todo = np.flatnonzero(np.isin(dets.role, ["person", "player"]))  # indices to classify
        feats, ok = jersey_colors(frame_rgb, dets.xyxy[todo])
        if len(ok) == 0:
            return labels
        idx = todo[ok]  # detection indices that have a colour feature
        dist = np.linalg.norm(feats[:, None, :] - self.centroids[None], axis=2)  # (M, k) distance to each cluster
        is_player = dets.role[idx] == "player"  # detected as "player" by a fine-tuned model
        dist[is_player, 2:] = np.inf  # a detected "player" can only be in a team, not a referee
        nearest = dist.argmin(axis=1)  # closest cluster per person
        far = (dist.min(axis=1) > self.outlier_distance) & ~is_player  # matches no kit -> other
        labels[idx] = np.where(far, OTHER, np.array(self.labels)[nearest])
        return labels

    def summary(self) -> dict:
        """Learned kit colours as a JSON-friendly dict (logged and saved to metrics.json)."""
        return {
            label: {"lab": [round(float(v), 1) for v in c], "hex": lab_to_hex(c), "calibration_samples": n}
            for label, c, n in zip(self.labels, self.centroids, self.sizes, strict=True)
        } | {"outlier_distance": self.outlier_distance}
