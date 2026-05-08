"""
Microscope Image Simulator for Microrobot Optical Tweezer
===========================================================
Generates synthetic microscope images that mimic the experimental
setup described in Ren et al. (MARSS 2022):
    - Microrobot A / B with spherical handles
    - Out-of-plane poses from 5° to 90°
    - Background noise, illumination variations
    - Multiple microrobots in one frame
    - Optical trap laser spots (optional)

This enables testing of detection and ellipse algorithms
without real experimental data.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random


@dataclass
class MicrorobotConfig:
    """Configuration for a microrobot appearance."""
    body_length: int = 40          # pixels
    body_width: int = 15
    sphere_radius: int = 12        # Spherical handle radius
    color: Tuple[int, int, int] = (180, 180, 190)
    opacity: float = 0.85


@dataclass
class SimulatedMicrorobot:
    """Instance of a microrobot in the simulated scene."""
    x: float
    y: float
    pose_angle: float = 0.0        # Out-of-plane pose in degrees (0-90)
    rotation: float = 0.0          # In-plane rotation
    robot_type: str = "A"
    config: Optional[MicrorobotConfig] = None

    def __post_init__(self):
        if self.config is None:
            if self.robot_type == "A":
                self.config = MicrorobotConfig(
                    body_length=45, body_width=16,
                    sphere_radius=14, color=(185, 185, 195)
                )
            else:
                self.config = MicrorobotConfig(
                    body_length=35, body_width=14,
                    sphere_radius=11, color=(175, 175, 185)
                )


class MicroscopeSimulator:
    """
    Simulates optical microscope images for microrobot experiments.
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (640, 480),
                 background_mean: int = 60,
                 background_std: int = 12,
                 noise_sigma: float = 8.0,
                 illumination_variation: bool = True):
        """
        Args:
            image_size: (width, height) of output images.
            background_mean: Mean gray value of background.
            background_std: Std of background texture.
            noise_sigma: Gaussian noise std added to final image.
            illumination_variation: Whether to add vignetting effect.
        """
        self.image_size = image_size
        self.background_mean = background_mean
        self.background_std = background_std
        self.noise_sigma = noise_sigma
        self.illumination_variation = illumination_variation

    def _generate_background(self) -> np.ndarray:
        """Generate textured microscope background."""
        w, h = self.image_size

        # Base background
        bg = np.ones((h, w), dtype=np.float32) * self.background_mean

        # Add low-frequency texture (simulates debris, uneven illumination)
        texture = np.random.normal(0, self.background_std, (h // 8, w // 8))
        texture = cv2.resize(texture, (w, h), interpolation=cv2.INTER_CUBIC)
        bg += texture

        # Add some random small particles (dust/debris)
        num_particles = random.randint(20, 60)
        for _ in range(num_particles):
            px = random.randint(0, w - 1)
            py = random.randint(0, h - 1)
            pr = random.randint(1, 3)
            intensity = random.randint(30, 80)
            cv2.circle(bg, (px, py), pr, intensity, -1)

        # Vignetting effect (dark corners)
        if self.illumination_variation:
            Y, X = np.ogrid[:h, :w]
            center_x, center_y = w / 2, h / 2
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
            vignette = 1.0 - 0.3 * (dist / max_dist) ** 2
            bg *= vignette

        return bg

    def _draw_spherical_handle(self, image: np.ndarray,
                                center: Tuple[float, float],
                                radius: float,
                                pose_angle: float,
                                color: Tuple[int, int, int],
                                opacity: float = 0.9) -> np.ndarray:
        """
        Draw a spherical handle with perspective foreshortening
        based on out-of-plane pose angle.
        """
        cx, cy = int(center[0]), int(center[1])

        # Foreshortening: as pose angle increases, sphere looks more elliptical
        # At 0°: circle, at 90°: highly flattened ellipse
        foreshortening = max(0.25, np.cos(np.deg2rad(pose_angle)))
        rx = int(radius)
        ry = int(max(radius * foreshortening, 3))

        overlay = image.copy()

        # Main ellipse body
        cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, color, -1)

        # Highlight (simulates spherical reflection)
        hx = cx - rx // 3
        hy = cy - ry // 3
        hr = max(2, int(min(rx, ry) * 0.25))
        highlight_color = tuple(min(255, c + 40) for c in color)
        cv2.circle(overlay, (hx, hy), hr, highlight_color, -1)

        # Shadow edge
        sx = cx + rx // 3
        sy = cy + ry // 3
        sr = max(2, int(min(rx, ry) * 0.3))
        shadow_color = tuple(max(0, c - 30) for c in color)
        cv2.ellipse(overlay, (sx, sy), (sr, int(sr * foreshortening)),
                    0, 0, 360, shadow_color, -1)

        # Blend
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        return image

    def _draw_microrobot_body(self, image: np.ndarray,
                               robot: SimulatedMicrorobot) -> np.ndarray:
        """Draw the full microrobot (body + spherical handles)."""
        cfg = robot.config
        if cfg is None:
            return image

        overlay = image.copy()
        cx, cy = int(robot.x), int(robot.y)
        angle = robot.rotation

        # Body length with foreshortening
        pose_rad = np.deg2rad(robot.pose_angle)
        body_len = cfg.body_length * max(0.3, np.cos(pose_rad))
        body_wid = cfg.body_width

        # Endpoints of body
        rad = np.deg2rad(angle)
        dx = np.cos(rad) * body_len / 2
        dy = np.sin(rad) * body_len / 2

        # Body as slightly blurred rectangle
        pts = np.array([
            [cx + dx + body_wid/2 * np.sin(rad), cy + dy - body_wid/2 * np.cos(rad)],
            [cx + dx - body_wid/2 * np.sin(rad), cy + dy + body_wid/2 * np.cos(rad)],
            [cx - dx - body_wid/2 * np.sin(rad), cy - dy + body_wid/2 * np.cos(rad)],
            [cx - dx + body_wid/2 * np.sin(rad), cy - dy - body_wid/2 * np.cos(rad)],
        ], dtype=np.int32)

        cv2.fillPoly(overlay, [pts], cfg.color)
        cv2.polylines(overlay, [pts], True,
                      tuple(max(0, c - 20) for c in cfg.color), 1)

        # Spherical handles at both ends
        end1 = (cx + dx, cy + dy)
        end2 = (cx - dx, cy - dy)

        overlay = self._draw_spherical_handle(
            overlay, end1, cfg.sphere_radius, robot.pose_angle,
            cfg.color, cfg.opacity
        )
        overlay = self._draw_spherical_handle(
            overlay, end2, cfg.sphere_radius, robot.pose_angle,
            cfg.color, cfg.opacity
        )

        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        return image

    def _add_laser_spots(self, image: np.ndarray,
                         spots: List[Tuple[float, float]],
                         spot_radius: int = 4) -> np.ndarray:
        """Draw optical trap laser spots as bright red circles."""
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        for sx, sy in spots:
            # Glow effect
            cv2.circle(vis, (int(sx), int(sy)), spot_radius + 3, (0, 0, 200), -1)
            cv2.circle(vis, (int(sx), int(sy)), spot_radius, (50, 50, 255), -1)
            cv2.circle(vis, (int(sx), int(sy)), spot_radius - 2, (150, 150, 255), -1)
        return vis

    def generate(self,
                 robots: List[SimulatedMicrorobot],
                 laser_spots: Optional[List[Tuple[float, float]]] = None,
                 return_annotations: bool = False,
                 background: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Generate a synthetic microscope image.

        Args:
            robots: List of microrobots to render.
            laser_spots: Optional list of laser spot positions.
            return_annotations: If True, return bounding boxes and labels.
            background: Optional pre-generated background to reuse (enables
                        static background across frames for video-like sequences).

        Returns:
            image: BGR image (uint8).
            annotations: Dict with 'boxes', 'labels', 'poses' if requested.
        """
        if background is not None:
            bg = background.copy()
        else:
            bg = self._generate_background()

        # Render microrobots
        for robot in robots:
            bg = self._draw_microrobot_body(bg, robot)

        # Convert to uint8
        image = np.clip(bg, 0, 255).astype(np.uint8)

        # Add Gaussian noise (kept per-frame for realism)
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Add slight blur to simulate microscope PSF
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)

        # Color conversion for consistency
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Add laser spots
        if laser_spots:
            image = self._add_laser_spots(image, laser_spots)

        annotations = None
        if return_annotations:
            annotations = self._compute_annotations(robots)

        return image, annotations

    def _compute_annotations(self,
                              robots: List[SimulatedMicrorobot]) -> Dict:
        """Compute ground-truth bounding boxes and properties."""
        boxes = []
        labels = []
        poses = []
        centers = []
        trap_points = []

        for r in robots:
            cfg = r.config
            if cfg is None:
                continue

            # Approximate bbox
            body_len = cfg.body_length * max(0.3, np.cos(np.deg2rad(r.pose_angle)))
            rad = np.deg2rad(r.rotation)
            dx = np.cos(rad) * body_len / 2
            dy = np.sin(rad) * body_len / 2

            x1 = min(r.x - dx, r.x + dx) - cfg.sphere_radius
            y1 = min(r.y - dy, r.y + dy) - cfg.sphere_radius
            x2 = max(r.x - dx, r.x + dx) + cfg.sphere_radius
            y2 = max(r.y - dy, r.y + dy) + cfg.sphere_radius

            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # Single class: microrobot
            poses.append(r.pose_angle)
            centers.append((r.x, r.y))

            # Trapping points = spherical handle centers
            trap_points.append((r.x + dx, r.y + dy))
            trap_points.append((r.x - dx, r.y - dy))

        return {
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'poses': np.array(poses, dtype=np.float32),
            'centers': centers,
            'trapping_points': trap_points,
        }

    # ------------------------------------------------------------------
    # Convenience generators
    # ------------------------------------------------------------------

    def generate_single(self,
                        pose_angle: float = 0.0,
                        rotation: float = 0.0,
                        robot_type: str = "A",
                        laser_spots: Optional[List[Tuple[float, float]]] = None,
                        return_annotations: bool = False):
        """Generate image with a single microrobot."""
        w, h = self.image_size
        robot = SimulatedMicrorobot(
            x=w / 2 + random.randint(-50, 50),
            y=h / 2 + random.randint(-50, 50),
            pose_angle=pose_angle,
            rotation=rotation,
            robot_type=robot_type
        )
        return self.generate([robot], laser_spots, return_annotations)

    def generate_multiple(self,
                          num_robots: int = 3,
                          pose_range: Tuple[float, float] = (5, 90),
                          return_annotations: bool = False):
        """Generate image with multiple microrobots."""
        w, h = self.image_size
        robots = []
        for _ in range(num_robots):
            margin = 80
            x = random.randint(margin, w - margin)
            y = random.randint(margin, h - margin)
            pose = random.uniform(*pose_range)
            rot = random.uniform(0, 180)
            rtype = random.choice(["A", "B"])
            robots.append(SimulatedMicrorobot(x, y, pose, rot, rtype))

        return self.generate(robots, laser_spots=None,
                             return_annotations=return_annotations)

    def generate_mosaic_batch(self,
                              batch_size: int = 4,
                              image_size: Tuple[int, int] = (416, 416)) -> np.ndarray:
        """
        Generate a Mosaic-augmented image (4 images stitched).
        Mirrors the data augmentation method described in the paper.
        """
        patches = []
        for _ in range(batch_size):
            sim = MicroscopeSimulator(image_size=image_size)
            img, ann = sim.generate_multiple(num_robots=random.randint(1, 3),
                                              return_annotations=True)

            # Random mirror
            if random.random() > 0.5:
                img = cv2.flip(img, 1)

            # Random crop to smaller size
            crop_h = random.randint(int(image_size[1] * 0.7), image_size[1])
            crop_w = random.randint(int(image_size[0] * 0.7), image_size[0])
            y0 = random.randint(0, image_size[1] - crop_h)
            x0 = random.randint(0, image_size[0] - crop_w)
            patch = img[y0:y0 + crop_h, x0:x0 + crop_w]
            patch = cv2.resize(patch, (image_size[0] // 2, image_size[1] // 2))
            patches.append(patch)

        # Stitch 2x2
        top = np.hstack([patches[0], patches[1]])
        bottom = np.hstack([patches[2], patches[3]])
        mosaic = np.vstack([top, bottom])
        mosaic = cv2.resize(mosaic, image_size)
        return mosaic


if __name__ == "__main__":
    sim = MicroscopeSimulator(image_size=(640, 480))

    # Single robot
    img, ann = sim.generate_single(pose_angle=30.0, rotation=45.0,
                                    return_annotations=True)
    cv2.imwrite("/tmp/sim_single.jpg", img)
    print("Single robot saved to /tmp/sim_single.jpg")
    if ann:
        print(f"Annotations: {ann['boxes'].shape[0]} boxes")

    # Multiple robots
    img, ann = sim.generate_multiple(num_robots=4, return_annotations=True)
    cv2.imwrite("/tmp/sim_multiple.jpg", img)
    print("Multiple robots saved to /tmp/sim_multiple.jpg")

    # Mosaic
    mosaic = sim.generate_mosaic_batch(batch_size=4)
    cv2.imwrite("/tmp/sim_mosaic.jpg", mosaic)
    print("Mosaic saved to /tmp/sim_mosaic.jpg")
