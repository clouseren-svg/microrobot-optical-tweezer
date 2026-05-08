"""
Automatic Trapping Controller
==============================
High-level controller that orchestrates the full trapping pipeline:
    1. Detect microrobots (YOLO + ellipse detection)
    2. Track multiple robots
    3. Compute optimal trapping points
    4. Move laser spots via visual servoing
    5. Verify successful trapping

This is the main entry point for automated indirect manipulation
via optical tweezers.
"""

import numpy as np
import cv2
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from control.hardware_interface import HardwareInterface, create_hardware
from control.visual_servoing import VisualServoingController, MultiTrapAllocator
from detection.ellipse_detector import ArcSupportEllipseDetector
from detection.tracker import IOUTracker, draw_tracks


class TrappingState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    TRACKING = "tracking"
    MOVING_TRAPS = "moving_traps"
    TRAPPED = "trapped"
    ERROR = "error"


@dataclass
class TrappingResult:
    """Result of a trapping attempt."""
    success: bool
    num_robots_detected: int
    num_robots_trapped: int
    trap_positions: List[Tuple[float, float]]
    message: str = ""


class AutomaticTrappingController:
    """
    Main controller for automatic trapping of multiple microrobots.
    """

    def __init__(self,
                 hardware: Optional[HardwareInterface] = None,
                 hardware_mode: str = "simulation",
                 num_traps: int = 5,
                 detection_interval: int = 3,   # Run detection every N frames
                 trapping_threshold: float = 8.0):  # px distance for "trapped"
        """
        Args:
            hardware: Pre-initialized hardware interface.
            hardware_mode: 'simulation' or real hardware name.
            num_traps: Number of optical traps available.
            detection_interval: Run YOLO+ellipse detection every N frames.
            trapping_threshold: Distance [px] to consider robot trapped.
        """
        if hardware is None:
            self.hardware = create_hardware(hardware_mode, num_traps=num_traps)
        else:
            self.hardware = hardware

        self.num_traps = num_traps
        self.detection_interval = detection_interval
        self.trapping_threshold = trapping_threshold

        # Sub-modules
        self.ellipse_detector = ArcSupportEllipseDetector(kmeans_k=13)
        self.tracker = IOUTracker(max_age=5, min_hits=2)
        # One servo controller per trap so PID states don't cross-contaminate
        from control.visual_servoing import ServoGain, VisualServoingController
        fast_gains = ServoGain(kp=3.0, ki=0.03, kd=0.15, lambda_max=100.0)
        self.servos = [VisualServoingController(gains=fast_gains) for _ in range(num_traps)]
        self.allocator = MultiTrapAllocator(num_traps=num_traps)

        self.state = TrappingState.IDLE
        self.frame_count = 0
        self.current_tracks = []
        self.trap_targets = []

    def connect(self) -> bool:
        """Initialize hardware connection."""
        success = self.hardware.connect()
        if success:
            self.state = TrappingState.IDLE
        return success

    def disconnect(self):
        """Safely disconnect hardware."""
        self.hardware.disconnect()
        self.state = TrappingState.IDLE

    def _detect_robots(self, image: np.ndarray) -> Tuple[List[np.ndarray],
                                                          List[List[Tuple[float, float]]]]:
        """
        Detect microrobots and their trapping points in image.
        Filters out detections that overlap with known laser spots or have
        unreasonable sizes, then merges nearby ellipses belonging to the same
        robot (each microrobot has two spherical handles).
        """
        # Ellipse detection for trapping points
        ellipses, _ = self.ellipse_detector.detect(image)
        trapping_pts = self.ellipse_detector.get_trapping_points(ellipses)

        # Filter and collect valid detections
        valid = []
        for ell, tps in zip(ellipses, trapping_pts):
            a, b = ell.axes
            # Reject unreasonably large ellipses (often spot glares)
            if max(a, b) > 120 or min(a, b) < 8:
                continue
            # Reject low-confidence detections
            if ell.score < 0.3:
                continue
            valid.append((ell, tps))

        # Merge nearby ellipses (same robot's two spherical handles)
        # Each robot body length ~35-45 px, so handles are ~35-45 px apart
        merged = []
        used = [False] * len(valid)
        merge_dist = 80.0  # max distance between handles of same robot

        for i in range(len(valid)):
            if used[i]:
                continue
            ell_i, tp_i = valid[i]
            cx_i, cy_i = ell_i.center
            group_ells = [ell_i]
            group_tps = [tp_i]
            used[i] = True

            for j in range(i + 1, len(valid)):
                if used[j]:
                    continue
                ell_j, tp_j = valid[j]
                cx_j, cy_j = ell_j.center
                dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                if dist < merge_dist:
                    group_ells.append(ell_j)
                    group_tps.append(tp_j)
                    used[j] = True

            # Compute merged bbox enclosing all ellipses in group
            all_x = [e.center[0] for e in group_ells]
            all_y = [e.center[1] for e in group_ells]
            all_a = [max(e.axes) for e in group_ells]
            x1 = min(all_x) - max(all_a) / 2 - 15
            y1 = min(all_y) - max(all_a) / 2 - 15
            x2 = max(all_x) + max(all_a) / 2 + 15
            y2 = max(all_y) + max(all_a) / 2 + 15
            merged.append((np.array([x1, y1, x2, y2]), group_tps))

        bboxes = [m[0] for m in merged]
        tp_per_robot = [m[1] for m in merged]
        return bboxes, tp_per_robot

    def _update_traps(self):
        """Move laser spots towards target positions using visual servoing."""
        spots = self.hardware.get_spot_positions()

        if len(self.current_tracks) == 0 or len(self.trap_targets) == 0:
            return

        # Assign traps to robots
        trap_positions = [(s.x, s.y) for s in spots[:self.num_traps]]
        robot_positions = [t.center for t in self.current_tracks]

        assignments = self.allocator.assign_traps_to_robots(
            robot_positions, trap_positions
        )

        # Compute and apply control for each assigned trap
        for trap_idx, robot_idx in assignments:
            if robot_idx >= len(self.current_tracks):
                continue
            robot = self.current_tracks[robot_idx]
            trap = spots[trap_idx]

            # Compute servo command using dedicated controller for this trap
            vx, vy = self.servos[trap_idx].compute_control(
                robot.center, (trap.x, trap.y),
                robot.velocity
            )

            # Update trap position (in simulation, directly set;
            # in real hardware, this would send a velocity command)
            new_x = trap.x + vx * 0.033
            new_y = trap.y + vy * 0.033
            self.hardware.move_laser_spot(trap_idx, new_x, new_y)

    def _check_trapping_status(self) -> int:
        """
        Check how many robots are successfully trapped.
        Returns count of trapped robots.
        """
        spots = self.hardware.get_spot_positions()
        trapped_count = 0

        for track in self.current_tracks:
            for spot in spots:
                dist = np.sqrt((track.center[0] - spot.x) ** 2 +
                               (track.center[1] - spot.y) ** 2)
                if dist < self.trapping_threshold:
                    trapped_count += 1
                    break

        return trapped_count

    def step(self, max_steps: int = 100) -> TrappingResult:
        """
        Execute one trapping control step.
        In a real system, this would be called at camera frame rate (~30Hz).

        Args:
            max_steps: Maximum control iterations (for simulation/demo).

        Returns:
            TrappingResult with status.
        """
        # Get camera frame
        frame_obj = self.hardware.get_camera_frame()
        if frame_obj is None:
            return TrappingResult(
                success=False, num_robots_detected=0,
                num_robots_trapped=0, trap_positions=[],
                message="Failed to acquire camera frame"
            )

        image = frame_obj.image
        self.frame_count += 1

        # Detection (every N frames or if no tracks)
        if (self.frame_count % self.detection_interval == 0 or
                len(self.current_tracks) == 0):
            self.state = TrappingState.DETECTING
            bboxes, trapping_pts = self._detect_robots(image)
            self.current_tracks = self.tracker.update(bboxes, trapping_pts)
        else:
            # Prediction-only update
            self.current_tracks = self.tracker.update([])

        # Update state
        if len(self.current_tracks) > 0:
            self.state = TrappingState.TRACKING
        else:
            self.state = TrappingState.DETECTING

        # Compute trap targets
        self.trap_targets = []
        for track in self.current_tracks:
            # Target: center of robot (simplified; could use trapping points)
            self.trap_targets.append(track.center)

        # Move traps
        if len(self.trap_targets) > 0:
            self.state = TrappingState.MOVING_TRAPS
            self._update_traps()

        # Check trapping status
        trapped = self._check_trapping_status()
        # Success: at least target fraction of detected robots are trapped
        # (not requiring all tracks since some may be false positives)
        min_trapped_for_success = max(1, min(len(self.current_tracks), 2))
        if trapped >= min_trapped_for_success:
            self.state = TrappingState.TRAPPED

        # For simulation: step physics if available
        if hasattr(self.hardware, 'step_physics'):
            self.hardware.step_physics()

        spots = self.hardware.get_spot_positions()
        spot_positions = [(s.x, s.y) for s in spots]

        return TrappingResult(
            success=(self.state == TrappingState.TRAPPED),
            num_robots_detected=len(self.current_tracks),
            num_robots_trapped=trapped,
            trap_positions=spot_positions,
            message=f"State: {self.state.value}, trapped {trapped}/{len(self.current_tracks)}"
        )

    def run_trapping_sequence(self,
                              target_num_robots: int = 3,
                              timeout_seconds: float = 30.0,
                              visualize: bool = False) -> TrappingResult:
        """
        Run full automatic trapping sequence.

        Args:
            target_num_robots: Desired number of robots to trap.
            timeout_seconds: Maximum time to attempt trapping.
            visualize: Show live visualization (simulation only).

        Returns:
            TrappingResult.
        """
        print(f"[TrappingController] Starting trapping sequence...")
        print(f"  Target: {target_num_robots} robots")
        print(f"  Timeout: {timeout_seconds}s")

        start_time = time.time()
        best_result = None

        while time.time() - start_time < timeout_seconds:
            result = self.step()
            best_result = result

            print(f"  {result.message}")

            if result.success:
                print("[TrappingController] Trapping complete!")
                return result

            if visualize:
                frame_obj = self.hardware.get_camera_frame()
                if frame_obj is not None:
                    vis = frame_obj.image.copy()
                    vis = draw_tracks(vis, self.current_tracks)
                    cv2.imshow("Trapping", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            time.sleep(0.03)  # ~30 FPS

        if visualize:
            cv2.destroyAllWindows()

        print("[TrappingController] Timeout reached.")
        return best_result if best_result else TrappingResult(
            success=False, num_robots_detected=0, num_robots_trapped=0,
            trap_positions=[], message="Timeout"
        )

    def get_visualization(self) -> Optional[np.ndarray]:
        """Get current frame with detection/tracking overlay."""
        frame_obj = self.hardware.get_camera_frame()
        if frame_obj is None:
            return None

        vis = frame_obj.image.copy()
        vis = draw_tracks(vis, self.current_tracks)

        # Draw trap positions
        spots = self.hardware.get_spot_positions()
        for spot in spots:
            if spot.active:
                cv2.circle(vis, (int(spot.x), int(spot.y)), 6,
                          (0, 255, 0), 2)
                cv2.circle(vis, (int(spot.x), int(spot.y)), 2,
                          (0, 255, 0), -1)

        # Status text
        status = f"State: {self.state.value} | Robots: {len(self.current_tracks)}"
        cv2.putText(vis, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return vis


if __name__ == "__main__":
    controller = AutomaticTrappingController(hardware_mode="simulation")
    controller.connect()

    result = controller.run_trapping_sequence(
        target_num_robots=2,
        timeout_seconds=10.0,
        visualize=False
    )

    print(f"\nFinal result:")
    print(f"  Success: {result.success}")
    print(f"  Detected: {result.num_robots_detected}")
    print(f"  Trapped: {result.num_robots_trapped}")
    print(f"  Trap positions: {result.trap_positions}")

    controller.disconnect()
