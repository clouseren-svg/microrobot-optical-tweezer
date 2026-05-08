"""
Multi-Microrobot Tracker
========================
Simple online real-time tracker for multiple microrobots.
Uses IoU-based association for high frame-rate tracking,
suitable for the real-time requirements of optical tweezer control.

Can be upgraded to DeepSORT for more robust tracking if needed.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Track:
    """A single track (trajectory) of a microrobot."""
    track_id: int
    bbox: np.ndarray              # [x1, y1, x2, y2]
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    trapping_points: List[Tuple[float, float]] = field(default_factory=list)
    is_confirmed: bool = False

    def __post_init__(self):
        self.history.append(self.center)

    def predict(self) -> "Track":
        """Predict next position using constant velocity model."""
        cx, cy = self.center
        vx, vy = self.velocity
        new_cx = cx + vx
        new_cy = cy + vy
        dx = (self.bbox[2] - self.bbox[0]) / 2
        dy = (self.bbox[3] - self.bbox[1]) / 2
        self.bbox = np.array([new_cx - dx, new_cy - dy,
                              new_cx + dx, new_cy + dy])
        self.center = (new_cx, new_cy)
        self.age += 1
        self.time_since_update += 1
        return self

    def update(self, bbox: np.ndarray,
               trapping_points: Optional[List[Tuple[float, float]]] = None):
        """Update track with new detection."""
        new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # Update velocity
        if len(self.history) > 0:
            old_center = self.history[-1]
            self.velocity = (new_center[0] - old_center[0],
                             new_center[1] - old_center[1])

        self.bbox = bbox
        self.center = new_center
        self.history.append(new_center)
        self.hits += 1
        self.time_since_update = 0

        if trapping_points is not None:
            self.trapping_points = trapping_points

        if self.hits >= 3:
            self.is_confirmed = True


class IOUTracker:
    """
    IoU-based multi-object tracker.
    Lightweight and suitable for real-time microrobot tracking.
    """

    def __init__(self,
                 max_age: int = 5,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Args:
            max_age: Maximum frames to keep a track without update.
            min_hits: Minimum hits to confirm a track.
            iou_threshold: Minimum IoU for matching.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1

    def _compute_iou(self, box_a: np.ndarray,
                     box_b: np.ndarray) -> float:
        """Compute IoU between two boxes [x1,y1,x2,y2]."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area + 1e-9

        return inter_area / union_area

    def _associate(self,
                   detections: List[np.ndarray]) -> Tuple[List[Tuple[int, int]],
                                                           List[int],
                                                           List[int]]:
        """
        Associate detections to existing tracks using IoU.
        Returns: (matched pairs, unmatched track indices, unmatched detection indices)
        """
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []

        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det)

        matched = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))

        # Greedy matching by highest IoU
        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < self.iou_threshold:
                break

            t_idx, d_idx = max_idx
            matched.append((t_idx, d_idx))
            unmatched_tracks.remove(t_idx)
            unmatched_dets.remove(d_idx)
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        return matched, unmatched_tracks, unmatched_dets

    def update(self,
               detections: List[np.ndarray],
               trapping_points_list: Optional[List[List[Tuple[float, float]]]] = None
               ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of bounding boxes [N, 4].
            trapping_points_list: Optional list of trapping points per detection.

        Returns:
            List of confirmed tracks.
        """
        if trapping_points_list is None:
            trapping_points_list = [[] for _ in detections]

        # Predict existing tracks
        for track in self.tracks:
            track.predict()

        # Associate
        matched, unmatched_tracks, unmatched_dets = self._associate(detections)

        # Update matched tracks
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx],
                                      trapping_points_list[d_idx])

        # Mark unmatched tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].time_since_update += 1

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            bbox = detections[d_idx]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            new_track = Track(
                track_id=self._next_id,
                bbox=bbox,
                center=center,
                trapping_points=trapping_points_list[d_idx]
            )
            self._next_id += 1
            self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        confirmed = [t for t in self.tracks
                     if t.is_confirmed or t.hits >= self.min_hits]
        return confirmed

    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed]

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self._next_id = 1


# ------------------------------------------------------------------
# Visualization helper
# ------------------------------------------------------------------

def draw_tracks(image: np.ndarray,
                tracks: List[Track],
                color_map: Optional[Dict[int, Tuple[int, int, int]]] = None
                ) -> np.ndarray:
    """Draw tracking results on image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    for track in tracks:
        tid = track.track_id
        if color_map and tid in color_map:
            color = color_map[tid]
        else:
            np.random.seed(tid)
            color = tuple(np.random.randint(50, 255, 3).tolist())

        x1, y1, x2, y2 = track.bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Track ID
        cv2.putText(vis, f"ID:{tid}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Trapping points
        for tp in track.trapping_points:
            tpx, tpy = int(tp[0]), int(tp[1])
            cv2.circle(vis, (tpx, tpy), 4, color, -1)
            cv2.circle(vis, (tpx, tpy), 6, (255, 255, 255), 1)

        # Trajectory history
        if len(track.history) > 1:
            pts = np.array([(int(p[0]), int(p[1])) for p in track.history],
                           dtype=np.int32)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, color, 1)

    return vis
