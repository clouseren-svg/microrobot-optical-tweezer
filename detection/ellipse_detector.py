"""
Ellipse Detection for Microrobot Trapping Points
=================================================
Implementation of the arc-support group based ellipse detection
method described in Ren et al. (MARSS 2022).

Four-step pipeline:
    1) Arcs Extraction      - Smoothing + Canny + contour extraction + corner splitting
    2) Arc-Support Groups   - Link line segments with convexity & continuity
    3) Ellipse Generation   - Fit ellipses from arc groups
    4) Filtering            - K-means clustering (k=13) to remove false positives

Reference for arc-support ellipse detection:
    Lu et al. "Arc-support line segments revisited: An efficient high-quality
    ellipse detection", IEEE TIP 2019.
    Wang et al. "Fast high-precision ellipse detection method", PR 2021.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EllipseCandidate:
    """Detected ellipse candidate."""
    center: Tuple[float, float]
    axes: Tuple[float, float]          # (major_axis, minor_axis)
    angle: float                        # Rotation angle in degrees
    score: float = 0.0                  # Salient score / confidence
    inliers: int = 0
    is_valid: bool = True


class ArcSupportEllipseDetector:
    """
    Ellipse detector using arc-support groups and k-means filtering.
    Optimized for detecting spherical/elliptical structures of microrobots
    in optical microscope images.
    """

    def __init__(self,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 min_arc_length: int = 15,
                 max_curvature_sigma: float = 3.0,
                 corner_alpha: float = 1.5,
                 min_salient_score: float = 0.3,
                 kmeans_k: int = 13,
                 kmeans_max_iter: int = 100,
                 min_ellipse_axis: int = 8,
                 max_ellipse_axis: int = 200,
                 ellipse_fit_threshold: float = 0.3):
        """
        Args:
            canny_low, canny_high: Canny edge detection thresholds.
            min_arc_length: Minimum arc length to consider.
            max_curvature_sigma: Gaussian sigma for curvature smoothing.
            corner_alpha: Threshold multiplier for corner detection.
            min_salient_score: Minimum score for arc-support groups.
            kmeans_k: Number of clusters for k-means filtering (paper uses k=13).
            min_ellipse_axis, max_ellipse_axis: Valid axis length range.
            ellipse_fit_threshold: Ransac-like inlier threshold.
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_arc_length = min_arc_length
        self.max_curvature_sigma = max_curvature_sigma
        self.corner_alpha = corner_alpha
        self.min_salient_score = min_salient_score
        self.kmeans_k = kmeans_k
        self.kmeans_max_iter = kmeans_max_iter
        self.min_ellipse_axis = min_ellipse_axis
        self.max_ellipse_axis = max_ellipse_axis
        self.ellipse_fit_threshold = ellipse_fit_threshold

    # ------------------------------------------------------------------
    # Step 1: Arcs Extraction
    # ------------------------------------------------------------------

    def extract_arcs(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract smooth arcs from image using:
        - Gaussian smoothing
        - Canny edge detection
        - 8-neighbor contour extraction
        - Curvature-based corner splitting
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gaussian smoothing
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Find contours with 8-neighbor connectivity
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

        arcs = []
        for cnt in contours:
            cnt = cnt.squeeze()
            if len(cnt.shape) != 2 or cnt.shape[0] < self.min_arc_length:
                continue

            # Split at corners based on curvature
            sub_arcs = self._split_at_corners(cnt)
            arcs.extend(sub_arcs)

        return arcs

    def _split_at_corners(self, contour: np.ndarray) -> List[np.ndarray]:
        """
        Split contour at corner points using curvature analysis.
        Uses Gaussian smoothing of curvature and adaptive threshold.
        """
        n = len(contour)
        if n < self.min_arc_length:
            return []

        # Compute curvature using finite differences
        # Closed curve handling
        pts = np.vstack([contour[-3:], contour, contour[:3]])

        curvature = np.zeros(n)
        for i in range(n):
            idx = i + 3
            x_m1, y_m1 = pts[idx - 1]
            x_0, y_0 = pts[idx]
            x_p1, y_p1 = pts[idx + 1]

            # First derivatives
            dx = (x_p1 - x_m1) / 2.0
            dy = (y_p1 - y_m1) / 2.0

            # Second derivatives
            ddx = x_p1 - 2 * x_0 + x_m1
            ddy = y_p1 - 2 * y_0 + y_m1

            denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
            k = abs(ddx * dy - dy * dx) / denom
            curvature[i] = k

        # Gaussian smoothing of curvature
        sigma = min(self.max_curvature_sigma, max(1.0, n / 50.0))
        ks = int(2 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        if ks >= 3:
            curvature_smooth = cv2.GaussianBlur(
                curvature.reshape(1, -1).astype(np.float32),
                (ks, 1), sigmaX=sigma
            ).flatten()
        else:
            curvature_smooth = curvature

        # Adaptive corner threshold
        kappa_max = curvature_smooth.max()
        threshold = self.corner_alpha * kappa_max

        # Find corner indices
        corner_mask = curvature_smooth > threshold
        corner_indices = np.where(corner_mask)[0]

        if len(corner_indices) == 0:
            return [contour]

        # Split contour at corners
        sub_arcs = []
        start = 0
        for ci in corner_indices:
            if ci - start >= self.min_arc_length:
                sub_arcs.append(contour[start:ci])
            start = ci

        # Last segment
        if n - start >= self.min_arc_length:
            sub_arcs.append(contour[start:])

        return sub_arcs

    # ------------------------------------------------------------------
    # Step 2 & 3: Arc-Support Groups and Ellipse Generation
    # ------------------------------------------------------------------

    def _fit_ellipse_to_arc(self, arc: np.ndarray) -> Optional[EllipseCandidate]:
        """Fit ellipse to a single arc using OpenCV."""
        if len(arc) < 6:
            return None

        try:
            ellipse = cv2.fitEllipse(arc.reshape(-1, 1, 2))
            (cx, cy), (ma, ma_minor), angle = ellipse

            # Filter by axis size
            if ma < self.min_ellipse_axis or ma > self.max_ellipse_axis:
                return None
            if ma_minor < self.min_ellipse_axis or ma_minor > self.max_ellipse_axis:
                return None

            # Aspect ratio check (not too eccentric)
            if ma_minor < 1 or ma / (ma_minor + 1e-6) > 5.0:
                return None

            # Check arc coverage (salient score proxy)
            # Compute how well the arc matches the fitted ellipse
            residuals = self._ellipse_residuals(arc, ellipse)
            inlier_ratio = np.mean(residuals < self.ellipse_fit_threshold)

            if inlier_ratio < 0.6:
                return None

            score = inlier_ratio * (len(arc) / 100.0)

            return EllipseCandidate(
                center=(cx, cy),
                axes=(ma, ma_minor),
                angle=angle,
                score=score,
                inliers=int(inlier_ratio * len(arc)),
                is_valid=True
            )
        except cv2.error:
            return None

    def _ellipse_residuals(self, points: np.ndarray,
                           ellipse: Tuple) -> np.ndarray:
        """
        Compute algebraic residuals of points to fitted ellipse.
        ellipse = ((cx, cy), (a, b), angle)
        """
        (cx, cy), (a, b), angle = ellipse
        theta = np.deg2rad(angle)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        dx = points[:, 0] - cx
        dy = points[:, 1] - cy

        # Rotate points to ellipse-aligned frame
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        # Algebraic distance
        a2 = (a / 2.0) ** 2 + 1e-9
        b2 = (b / 2.0) ** 2 + 1e-9
        dist = (x_rot ** 2 / a2 + y_rot ** 2 / b2 - 1.0)
        return np.abs(dist)

    def generate_ellipse_candidates(self,
                                     arcs: List[np.ndarray]) -> List[EllipseCandidate]:
        """
        Generate ellipse candidates from arcs.
        Includes both single-arc fitting and paired-arc fitting.
        """
        candidates = []

        # Single arc fitting
        for arc in arcs:
            if len(arc) < 6:
                continue
            ell = self._fit_ellipse_to_arc(arc)
            if ell is not None and ell.score >= self.min_salient_score:
                candidates.append(ell)

        # Paired arc fitting (complementary arcs)
        n = len(arcs)
        for i in range(n):
            for j in range(i + 1, n):
                combined = np.vstack([arcs[i], arcs[j]])
                if len(combined) < 10:
                    continue

                # Check convexity/continuity condition (simplified)
                # In full implementation, check arc-support line segment linking
                ell = self._fit_ellipse_to_arc(combined)
                if ell is not None and ell.score >= self.min_salient_score:
                    candidates.append(ell)

        return candidates

    # ------------------------------------------------------------------
    # Step 4: Filtering by K-Means Clustering
    # ------------------------------------------------------------------

    def _nms_ellipses(self,
                      candidates: List[EllipseCandidate],
                      center_iou_thresh: float = 0.3) -> List[EllipseCandidate]:
        """
        Non-maximum suppression for ellipses based on center proximity.
        Keeps the highest-scoring ellipse when multiple detections overlap.
        """
        if len(candidates) <= 1:
            return candidates

        # Sort by score descending
        sorted_cands = sorted(candidates, key=lambda e: e.score, reverse=True)
        keep = []

        for cand in sorted_cands:
            is_duplicate = False
            for kept in keep:
                # Compute center distance relative to average axis size
                dist = np.sqrt((cand.center[0] - kept.center[0])**2 +
                               (cand.center[1] - kept.center[1])**2)
                avg_axis = (kept.axes[0] + kept.axes[1]) / 4.0 + 1e-6
                if dist < avg_axis * center_iou_thresh * 2:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(cand)

        return keep

    def filter_by_kmeans(self,
                         candidates: List[EllipseCandidate]) -> List[EllipseCandidate]:
        """
        Use k-means clustering (k=13, as in paper) to cluster ellipse centers
        and remove outliers / false positives.
        """
        if len(candidates) == 0:
            return []

        if len(candidates) <= self.kmeans_k:
            # Not enough candidates for k-means; return all valid ones
            return sorted(candidates, key=lambda e: e.score, reverse=True)

        # Prepare features: [center_x, center_y, axis_major, axis_minor, angle]
        features = np.array([
            [e.center[0], e.center[1], e.axes[0], e.axes[1], e.angle]
            for e in candidates
        ])

        # Normalize features
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0) + 1e-9
        features_norm = (features - feat_mean) / feat_std

        # K-means clustering
        kmeans = KMeans(n_clusters=self.kmeans_k,
                        max_iter=self.kmeans_max_iter,
                        n_init=10,
                        random_state=42)
        labels = kmeans.fit_predict(features_norm)

        # Select representative ellipses from each cluster
        # Prefer high-score candidates near cluster center
        filtered = []
        for k in range(self.kmeans_k):
            mask = labels == k
            cluster_indices = np.where(mask)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_features = features_norm[mask]
            cluster_center = kmeans.cluster_centers_[k]

            # Compute distance to cluster center
            dists = np.linalg.norm(cluster_features - cluster_center, axis=1)

            # Weight by score and proximity to center
            scores = np.array([candidates[i].score for i in cluster_indices])
            weights = scores / (dists + 0.1)

            best_idx = cluster_indices[np.argmax(weights)]
            filtered.append(candidates[best_idx])

        # Sort by score
        filtered = sorted(filtered, key=lambda e: e.score, reverse=True)
        return filtered

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray,
               return_debug: bool = False) -> Tuple[List[EllipseCandidate],
                                                     Optional[np.ndarray]]:
        """
        Main detection pipeline.

        Args:
            image: Input image (BGR or grayscale).
            return_debug: If True, return debug visualization.

        Returns:
            (list of EllipseCandidate, debug_image or None)
        """
        # Step 1: Extract arcs
        arcs = self.extract_arcs(image)

        # Steps 2 & 3: Generate ellipse candidates
        candidates = self.generate_ellipse_candidates(arcs)

        # Step 3.5: NMS to remove duplicate/overlapping detections
        candidates = self._nms_ellipses(candidates)

        # Step 4: Filter by k-means
        filtered = self.filter_by_kmeans(candidates)

        debug_img = None
        if return_debug:
            debug_img = self._draw_debug(image, arcs, candidates, filtered)

        return filtered, debug_img

    def _draw_debug(self, image: np.ndarray,
                    arcs: List[np.ndarray],
                    candidates: List[EllipseCandidate],
                    filtered: List[EllipseCandidate]) -> np.ndarray:
        """Draw debug visualization."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        h, w = vis.shape[:2]
        scale = 800 / max(h, w)
        if scale < 1.0:
            vis = cv2.resize(vis, None, fx=scale, fy=scale)

        # Draw all arcs in light green
        for arc in arcs:
            if len(arc) > 1:
                pts = (arc * scale).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, (0, 255, 100), 1)

        # Draw all candidates in yellow (thin)
        for ell in candidates:
            c = (int(ell.center[0] * scale), int(ell.center[1] * scale))
            a = (int(ell.axes[0] * scale / 2), int(ell.axes[1] * scale / 2))
            cv2.ellipse(vis, c, a, ell.angle, 0, 360, (0, 255, 255), 1)

        # Draw filtered ellipses in red (thick)
        for ell in filtered:
            c = (int(ell.center[0] * scale), int(ell.center[1] * scale))
            a = (int(ell.axes[0] * scale / 2), int(ell.axes[1] * scale / 2))
            cv2.ellipse(vis, c, a, ell.angle, 0, 360, (0, 0, 255), 2)
            cv2.circle(vis, c, 3, (0, 0, 255), -1)

        return vis

    def get_trapping_points(self, ellipses: List[EllipseCandidate],
                            num_points_per_ellipse: int = 1) -> List[Tuple[float, float]]:
        """
        Get optimal trapping points from detected ellipses.
        By default, returns the center of each ellipse as the trapping point.
        """
        points = []
        for ell in ellipses:
            if num_points_per_ellipse == 1:
                points.append(ell.center)
            else:
                # Multiple points around the ellipse
                for k in range(num_points_per_ellipse):
                    angle = 2 * np.pi * k / num_points_per_ellipse
                    dx = (ell.axes[0] / 2) * np.cos(angle)
                    dy = (ell.axes[1] / 2) * np.sin(angle)
                    rad = np.deg2rad(ell.angle)
                    rx = dx * np.cos(rad) - dy * np.sin(rad)
                    ry = dx * np.sin(rad) + dy * np.cos(rad)
                    points.append((ell.center[0] + rx, ell.center[1] + ry))
        return points


# ------------------------------------------------------------------
# Standalone function interface
# ------------------------------------------------------------------

def detect_ellipses(image: np.ndarray,
                    canny_low: int = 50,
                    canny_high: int = 150,
                    kmeans_k: int = 13,
                    return_debug: bool = False):
    """
    Convenience function for ellipse detection.

    Returns:
        ellipses: List of EllipseCandidate
        trapping_points: List of (x, y) center points
        debug_image: Optional debug visualization
    """
    detector = ArcSupportEllipseDetector(
        canny_low=canny_low,
        canny_high=canny_high,
        kmeans_k=kmeans_k
    )
    ellipses, debug_img = detector.detect(image, return_debug=return_debug)
    trapping_points = detector.get_trapping_points(ellipses)
    return ellipses, trapping_points, debug_img


if __name__ == "__main__":
    # Test with a synthetic ellipse image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.ellipse(img, (200, 200), (80, 60), 30, 0, 360, (200, 200, 200), -1)
    cv2.ellipse(img, (100, 100), (40, 30), 0, 0, 360, (180, 180, 180), -1)

    # Add noise
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    detector = ArcSupportEllipseDetector(kmeans_k=2)
    ellipses, debug = detector.detect(img, return_debug=True)

    print(f"Detected {len(ellipses)} ellipses:")
    for i, e in enumerate(ellipses):
        print(f"  Ellipse {i+1}: center={e.center}, axes={e.axes}, "
              f"angle={e.angle:.1f}°, score={e.score:.3f}")

    if debug is not None:
        cv2.imwrite("/tmp/ellipse_debug.jpg", debug)
        print("Debug image saved to /tmp/ellipse_debug.jpg")
