"""
Optical Tweezer Physics Simulation
===================================
Simplified physical model of optical trapping for software-in-the-loop
testing. Based on the multi-virtual-spring model described in
Zhang et al. "Distributed force control for microrobot manipulation
via planar multi-spot optical tweezer" (Adv. Opt. Mater. 2020).

Assumptions:
- Geometrical optics regime (particle >> wavelength)
- Rigid body microrobot
- Negligible gravity/buoyancy when trapped
- Gaussian beam profile
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trap:
    """Optical trap (laser spot)."""
    x: float
    y: float
    power: float = 1.0          # Relative laser power [0, 1]
    active: bool = True


@dataclass
class MicrorobotState:
    """2D state of a microrobot."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0          # Orientation [rad]
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0          # Angular velocity

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta,
                         self.vx, self.vy, self.omega])


class OpticalTweezerSimulator:
    """
    Simulates optical trapping forces on microrobots.
    Uses a simplified virtual spring model:
        F = -k * (r - r_trap) * exp(-||r - r_trap||^2 / (2*w^2))
    where k is stiffness, w is beam waist.
    """

    def __init__(self,
                 trap_stiffness: float = 0.5,      # pN/nm (scaled units)
                 trap_range: float = 50.0,          # Effective trapping range [px]
                 beam_waist: float = 5.0,           # Beam waist [px]
                 damping: float = 0.9,              # Viscous damping factor
                 dt: float = 0.01,                  # Simulation timestep [s]
                 mass: float = 1.0):                # Normalized mass
        self.k = trap_stiffness
        self.trap_range = trap_range
        self.w = beam_waist
        self.damping = damping
        self.dt = dt
        self.mass = mass

    def compute_trap_force(self,
                           robot: MicrorobotState,
                           trap: Trap) -> Tuple[float, float]:
        """
        Compute optical trapping force from a single trap.
        Returns (Fx, Fy) in normalized force units.
        """
        if not trap.active:
            return 0.0, 0.0

        dx = robot.x - trap.x
        dy = robot.y - trap.y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist > self.trap_range:
            return 0.0, 0.0

        # Gaussian intensity profile -> spring-like restoring force
        intensity = np.exp(-dist ** 2 / (2 * self.w ** 2))
        force_mag = -self.k * dist * intensity * trap.power

        if dist < 1e-6:
            return 0.0, 0.0

        fx = force_mag * (dx / dist)
        fy = force_mag * (dy / dist)
        return fx, fy

    def compute_total_force(self,
                            robot: MicrorobotState,
                            traps: List[Trap]) -> Tuple[float, float, float]:
        """
        Compute total force and torque from all active traps.
        Returns (Fx, Fy, torque).
        """
        fx_total, fy_total = 0.0, 0.0
        torque = 0.0

        for trap in traps:
            fx, fy = self.compute_trap_force(robot, trap)
            fx_total += fx
            fy_total += fy

            # Torque = r x F
            dx = trap.x - robot.x
            dy = trap.y - robot.y
            torque += dx * fy - dy * fx

        return fx_total, fy_total, torque

    def step(self,
             robot: MicrorobotState,
             traps: List[Trap]) -> MicrorobotState:
        """
        Integrate one simulation step using Euler method.
        """
        fx, fy, torque = self.compute_total_force(robot, traps)

        # Acceleration
        ax = fx / self.mass
        ay = fy / self.mass
        alpha = torque / (self.mass * 10.0)  # Approximate moment of inertia

        # Update velocities (with damping)
        robot.vx = (robot.vx + ax * self.dt) * self.damping
        robot.vy = (robot.vy + ay * self.dt) * self.damping
        robot.omega = (robot.omega + alpha * self.dt) * self.damping

        # Update positions
        robot.x += robot.vx * self.dt
        robot.y += robot.vy * self.dt
        robot.theta += robot.omega * self.dt

        return robot

    def simulate_trapping(self,
                          initial_state: MicrorobotState,
                          trap_positions: List[Tuple[float, float]],
                          steps: int = 500) -> List[MicrorobotState]:
        """
        Simulate trapping a microrobot with given trap positions.
        Returns trajectory of states.
        """
        traps = [Trap(x=x, y=y) for x, y in trap_positions]
        robot = MicrorobotState(
            x=initial_state.x,
            y=initial_state.y,
            theta=initial_state.theta,
            vx=initial_state.vx,
            vy=initial_state.vy,
            omega=initial_state.omega
        )

        trajectory = [robot]
        for _ in range(steps):
            robot = self.step(robot, traps)
            trajectory.append(MicrorobotState(
                x=robot.x, y=robot.y, theta=robot.theta,
                vx=robot.vx, vy=robot.vy, omega=robot.omega
            ))

        return trajectory

    def is_trapped(self,
                   robot: MicrorobotState,
                   trap: Trap,
                   threshold: float = 3.0) -> bool:
        """Check if robot is stably trapped near a trap."""
        dist = np.sqrt((robot.x - trap.x) ** 2 + (robot.y - trap.y) ** 2)
        speed = np.sqrt(robot.vx ** 2 + robot.vy ** 2)
        return dist < threshold and speed < 1.0


# ------------------------------------------------------------------
# Multi-robot trapping simulation
# ------------------------------------------------------------------

class MultiRobotTrappingSim:
    """Simulate simultaneous trapping of multiple microrobots."""

    def __init__(self, num_robots: int = 3, **sim_kwargs):
        self.num_robots = num_robots
        self.ot_sim = OpticalTweezerSimulator(**sim_kwargs)
        self.robots: List[MicrorobotState] = []
        self.traps: List[List[Trap]] = []

    def reset(self, initial_positions: Optional[List[Tuple[float, float]]] = None):
        """Initialize robots at given positions."""
        if initial_positions is None:
            # Random positions
            self.robots = [
                MicrorobotState(x=np.random.uniform(100, 540),
                               y=np.random.uniform(100, 380))
                for _ in range(self.num_robots)
            ]
        else:
            self.robots = [
                MicrorobotState(x=x, y=y) for x, y in initial_positions
            ]
        self.traps = [[] for _ in range(self.num_robots)]

    def assign_traps(self,
                     robot_idx: int,
                     trap_positions: List[Tuple[float, float]]):
        """Assign trap positions to a specific robot."""
        self.traps[robot_idx] = [Trap(x=x, y=y) for x, y in trap_positions]

    def step_all(self) -> List[MicrorobotState]:
        """Advance simulation for all robots."""
        new_states = []
        for i, robot in enumerate(self.robots):
            new_robot = self.ot_sim.step(robot, self.traps[i])
            new_states.append(new_robot)
        self.robots = new_states
        return new_states

    def run(self, steps: int = 300) -> List[List[MicrorobotState]]:
        """Run full simulation, return trajectories for all robots."""
        trajectories = [[] for _ in range(self.num_robots)]
        for _ in range(steps):
            states = self.step_all()
            for i, s in enumerate(states):
                trajectories[i].append(MicrorobotState(
                    x=s.x, y=s.y, theta=s.theta,
                    vx=s.vx, vy=s.vy, omega=s.omega
                ))
        return trajectories


if __name__ == "__main__":
    sim = OpticalTweezerSimulator()

    # Test single trapping
    robot = MicrorobotState(x=200, y=200)
    traj = sim.simulate_trapping(robot, [(250, 250)], steps=500)

    final = traj[-1]
    print(f"Trapping simulation:")
    print(f"  Initial: ({traj[0].x:.1f}, {traj[0].y:.1f})")
    print(f"  Final:   ({final.x:.1f}, {final.y:.1f})")
    print(f"  Distance to trap: {np.sqrt((final.x-250)**2 + (final.y-250)**2):.2f} px")

    # Multi-robot
    multi = MultiRobotTrappingSim(num_robots=2)
    multi.reset([(100, 100), (400, 300)])
    multi.assign_traps(0, [(150, 150)])
    multi.assign_traps(1, [(450, 350)])
    trajs = multi.run(steps=300)
    print(f"\nMulti-robot trapping:")
    for i, t in enumerate(trajs):
        print(f"  Robot {i+1}: final pos = ({t[-1].x:.1f}, {t[-1].y:.1f})")
