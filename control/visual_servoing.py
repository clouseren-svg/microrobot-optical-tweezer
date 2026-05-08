"""
Visual Servoing for Microrobot Manipulation
============================================
Implements image-based visual servoing (IBVS) for controlling
laser spot positions to trap and manipulate microrobots.

Control law:
    u = -lambda * L^+ * e
where:
    e = s - s*   (feature error)
    L = interaction matrix
    s = [x_trap - x_robot, y_trap - y_robot]
    s* = [0, 0]  (desired: trap centered on robot)
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ServoGain:
    """PID-like gains for visual servoing."""
    kp: float = 0.5       # Proportional
    ki: float = 0.01      # Integral
    kd: float = 0.1       # Derivative
    lambda_max: float = 10.0  # Max control output [px/step]


class VisualServoingController:
    """
    Image-based visual servoing controller for optical tweezer.
    Maps feature errors (robot-trap position difference) to
    laser spot velocity commands.
    """

    def __init__(self, gains: Optional[ServoGain] = None,
                 dt: float = 0.033):  # ~30 FPS
        self.gains = gains or ServoGain()
        self.dt = dt
        self._error_integral = np.zeros(2)
        self._error_prev = np.zeros(2)

    def compute_control(self,
                        robot_pos: Tuple[float, float],
                        trap_pos: Tuple[float, float],
                        robot_vel: Tuple[float, float] = (0.0, 0.0)
                        ) -> Tuple[float, float]:
        """
        Compute control command to move trap towards robot.

        Args:
            robot_pos: Current robot center (x, y) [px].
            trap_pos: Current trap position (x, y) [px].
            robot_vel: Estimated robot velocity (vx, vy) [px/s].

        Returns:
            (trap_vx, trap_vy): Trap velocity command [px/s].
        """
        # Feature error: trap should be at robot position
        error = np.array([
            robot_pos[0] - trap_pos[0],
            robot_pos[1] - trap_pos[1]
        ], dtype=np.float32)

        # Proportional
        p_term = self.gains.kp * error

        # Integral
        self._error_integral += error * self.dt
        self._error_integral = np.clip(self._error_integral, -100, 100)
        i_term = self.gains.ki * self._error_integral

        # Derivative
        d_error = (error - self._error_prev) / self.dt
        d_term = self.gains.kd * d_error
        self._error_prev = error.copy()

        # Feedforward: move trap to intercept moving robot
        feedforward = np.array(robot_vel)

        # Total control
        control = p_term + i_term + d_term + feedforward

        # Clamp to max speed
        norm = np.linalg.norm(control)
        if norm > self.gains.lambda_max:
            control = control / norm * self.gains.lambda_max

        return float(control[0]), float(control[1])

    def reset(self):
        """Reset integral and derivative memory."""
        self._error_integral = np.zeros(2)
        self._error_prev = np.zeros(2)


class MultiTrapAllocator:
    """
    Allocates multiple optical traps to multiple microrobots.
    Uses a greedy assignment based on nearest-neighbor matching.
    """

    def __init__(self, num_traps: int = 5):
        self.num_traps = num_traps

    def assign_traps_to_robots(
        self,
        robot_positions: List[Tuple[float, float]],
        trap_positions: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Assign each trap to the nearest robot.
        Returns list of (trap_idx, robot_idx) pairs.
        """
        assignments = []
        available_traps = list(range(len(trap_positions)))
        available_robots = list(range(len(robot_positions)))

        while available_traps and available_robots:
            best_dist = float('inf')
            best_pair = None

            for ti in available_traps:
                for ri in available_robots:
                    tx, ty = trap_positions[ti]
                    rx, ry = robot_positions[ri]
                    dist = np.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (ti, ri)

            if best_pair:
                assignments.append(best_pair)
                available_traps.remove(best_pair[0])
                available_robots.remove(best_pair[1])
            else:
                break

        return assignments

    def compute_trap_targets(
        self,
        robot_positions: List[Tuple[float, float]],
        robot_orientations: List[float],
        trapping_offsets: List[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """
        Compute target positions for traps based on robot trapping points.

        Args:
            robot_positions: List of (x, y) robot centers.
            robot_orientations: List of orientation angles [deg].
            trapping_offsets: List of offset vectors per robot
                             (relative to center in robot frame).

        Returns:
            List of absolute trap target positions.
        """
        targets = []
        for (cx, cy), theta, offsets in zip(
            robot_positions, robot_orientations, trapping_offsets
        ):
            rad = np.deg2rad(theta)
            cos_t = np.cos(rad)
            sin_t = np.sin(rad)

            for dx, dy in offsets:
                # Rotate offset by robot orientation
                rx = dx * cos_t - dy * sin_t
                ry = dx * sin_t + dy * cos_t
                targets.append((cx + rx, cy + ry))

        return targets


if __name__ == "__main__":
    # Test visual servoing
    servo = VisualServoingController()

    robot = (300.0, 250.0)
    trap = (280.0, 220.0)

    for step in range(20):
        vx, vy = servo.compute_control(robot, trap)
        trap = (trap[0] + vx * 0.033, trap[1] + vy * 0.033)
        dist = np.sqrt((robot[0] - trap[0])**2 + (robot[1] - trap[1])**2)
        print(f"Step {step+1}: trap=({trap[0]:.1f}, {trap[1]:.1f}), dist={dist:.2f}")

    # Test allocator
    allocator = MultiTrapAllocator(num_traps=3)
    robots = [(100, 100), (300, 200), (500, 400)]
    traps = [(110, 110), (290, 210), (490, 390), (200, 300)]
    assigned = allocator.assign_traps_to_robots(robots, traps)
    print(f"\nTrap assignments: {assigned}")
