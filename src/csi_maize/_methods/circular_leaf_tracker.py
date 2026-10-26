"""Prediction-only association with explicit nominal rotation; no teacher IDs."""

from multimodal_leaf_core import assign_tracks, mask_iou
import numpy as np


def harmonic_predict(views, xy, next_view):
    a = np.asarray(views) * 2 * np.pi / 40
    design = np.column_stack([np.ones(len(a)), np.cos(a), np.sin(a)])
    coef = np.linalg.lstsq(design, np.asarray(xy), rcond=None)[0]
    b = next_view * 2 * np.pi / 40
    return np.array([1, np.cos(b), np.sin(b)]) @ coef


class CircularTracker:
    def __init__(self, mode="circular", max_gap=8):
        self.mode = mode
        self.max_gap = max_gap
        self.tracks = {}
        self.counter = 0

    def step(self, view, observations):
        active = [
            k for k, t in self.tracks.items() if abs(view - t["views"][-1]) <= self.max_gap + 1
        ]
        cost = np.full((len(active), len(observations)), 9.0)
        for i, k in enumerate(active):
            t = self.tracks[k]
            prediction = t["centroids"][-1]
            if self.mode == "circular" and len(t["views"]) >= 6 and np.ptp(t["views"]) >= 8:
                prediction = harmonic_predict(t["views"], t["centroids"], view)
            elif self.mode == "circular" and len(t["views"]) >= 2:
                dv = t["views"][-1] - t["views"][-2]
                prediction = (
                    np.array(prediction)
                    + (view - t["views"][-1])
                    * (np.array(t["centroids"][-1]) - t["centroids"][-2])
                    / dv
                )
            for j, o in enumerate(observations):
                overlap = mask_iou(t["mask"], o["mask"])
                distance = np.linalg.norm(np.asarray(o["centroid"]) - prediction) / 65
                area = min(2.0, abs(np.log((o["area"] + 1) / (t["areas"][-1] + 1))))
                cost[i, j] = (
                    (1 - overlap)
                    if self.mode == "overlap"
                    else 0.45 * (1 - overlap) + 0.45 * distance + 0.1 * area
                )
        pairs = assign_tracks(cost, 0.85 if self.mode == "overlap" else 1.0)
        result = [None] * len(observations)
        for i, j in pairs:
            result[j] = active[i]
        for j, o in enumerate(observations):
            if result[j] is None:
                self.counter += 1
                result[j] = self.counter
                self.tracks[self.counter] = {"views": [], "centroids": [], "areas": []}
            t = self.tracks[result[j]]
            t["views"].append(view)
            t["centroids"].append(list(o["centroid"]))
            t["areas"].append(o["area"])
            t["mask"] = o["mask"]
        assert len(set(result)) == len(result)
        return result
