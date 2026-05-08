"""
Hardware Abstraction Layer for Optical Tweezer System
=======================================================
Provides a unified interface for controlling optical tweezer hardware.
Supports both simulation mode (software-in-the-loop) and real hardware.

To use with real hardware, subclass HardwareInterface and implement:
    - move_laser_spot(spot_id, x, y)
    - set_laser_power(spot_id, power)
    - get_camera_frame()
    - get_spot_positions()

Reference hardware: Elliot Scientific E3500 OT system + Basler CCD camera.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time


@dataclass
class LaserSpot:
    """State of a single laser spot."""
    spot_id: int
    x: float
    y: float
    power: float = 1.0
    active: bool = True


@dataclass
class CameraFrame:
    """Camera frame metadata."""
    timestamp: float
    image: np.ndarray
    exposure_us: float = 10000.0
    gain_db: float = 0.0


class HardwareInterface(ABC):
    """
    Abstract base class for optical tweezer hardware control.
    All real hardware implementations must subclass this.
    """

    def __init__(self, image_size: Tuple[int, int] = (640, 480),
                 num_traps: int = 5):
        self.image_size = image_size
        self.num_traps = num_traps
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Initialize hardware connection."""
        pass

    @abstractmethod
    def disconnect(self):
        """Safely shut down hardware."""
        pass

    @abstractmethod
    def move_laser_spot(self, spot_id: int, x: float, y: float) -> bool:
        """
        Move a laser spot to (x, y) in image coordinates [pixels].
        Returns True if command was accepted.
        """
        pass

    @abstractmethod
    def set_laser_power(self, spot_id: int, power: float) -> bool:
        """
        Set laser power for a spot [0.0, 1.0].
        """
        pass

    @abstractmethod
    def get_camera_frame(self) -> Optional[CameraFrame]:
        """Acquire a single frame from the microscope camera."""
        pass

    @abstractmethod
    def get_spot_positions(self) -> List[LaserSpot]:
        """Get current positions of all laser spots."""
        pass

    def move_multiple_spots(self,
                            spots: List[Tuple[int, float, float]]) -> bool:
        """Batch move multiple spots."""
        success = True
        for spot_id, x, y in spots:
            success = success and self.move_laser_spot(spot_id, x, y)
        return success

    def home_all_spots(self):
        """Move all spots to center of field of view."""
        cx, cy = self.image_size[0] / 2, self.image_size[1] / 2
        for i in range(self.num_traps):
            self.move_laser_spot(i, cx, cy)


# ------------------------------------------------------------------
# Simulation Hardware (for testing without real equipment)
# ------------------------------------------------------------------

class SimulationHardware(HardwareInterface):
    """
    Software-only hardware interface for development and testing.
    Simulates camera images and laser spot movements using the
    microscope simulator and optical tweezer physics model.
    """

    def __init__(self, image_size: Tuple[int, int] = (640, 480),
                 num_traps: int = 5,
                 frame_rate: float = 30.0):
        super().__init__(image_size, num_traps)
        self.frame_rate = frame_rate
        self._spots: Dict[int, LaserSpot] = {
            i: LaserSpot(i, image_size[0] / 2, image_size[1] / 2)
            for i in range(num_traps)
        }
        self._sim_time = 0.0
        self._last_frame_time = 0.0

        # Import simulators
        from simulation.microscope_simulator import MicroscopeSimulator, SimulatedMicrorobot
        from simulation.optical_tweezer_sim import OpticalTweezerSimulator, MicrorobotState

        self._microscope = MicroscopeSimulator(image_size=image_size)
        self._ot_sim = OpticalTweezerSimulator()
        self._simulated_robots: List[SimulatedMicrorobot] = []
        self._robot_states: List[MicrorobotState] = []
        self._cached_background: Optional[np.ndarray] = None

        self._setup_scene()

    def _setup_scene(self):
        """Initialize simulated scene with some microrobots."""
        from simulation.microscope_simulator import SimulatedMicrorobot
        from simulation.optical_tweezer_sim import MicrorobotState

        w, h = self.image_size
        self._simulated_robots = [
            SimulatedMicrorobot(x=w * 0.3, y=h * 0.4,
                                pose_angle=15.0, rotation=30.0, robot_type="A"),
            SimulatedMicrorobot(x=w * 0.6, y=h * 0.5,
                                pose_angle=45.0, rotation=-20.0, robot_type="B"),
            SimulatedMicrorobot(x=w * 0.5, y=h * 0.7,
                                pose_angle=30.0, rotation=60.0, robot_type="A"),
        ]
        self._robot_states = [
            MicrorobotState(x=r.x, y=r.y) for r in self._simulated_robots
        ]

        # Initialize traps scattered around the scene, avoiding direct overlap
        # with robots but close enough for reasonable trapping times
        init_positions = [
            (w * 0.25, h * 0.30),
            (w * 0.75, h * 0.30),
            (w * 0.25, h * 0.70),
            (w * 0.75, h * 0.70),
            (w * 0.50, h * 0.50),
        ]
        for i in range(self.num_traps):
            x, y = init_positions[i % len(init_positions)]
            self._spots[i] = LaserSpot(i, x, y, power=1.0, active=True)

    def connect(self) -> bool:
        self.is_connected = True
        # Pre-generate a static background so detections are stable across frames
        self._cached_background = self._microscope._generate_background()
        print("[SimHW] Connected to simulated OT system")
        return True

    def disconnect(self):
        self.is_connected = False
        print("[SimHW] Disconnected")

    def move_laser_spot(self, spot_id: int, x: float, y: float) -> bool:
        if spot_id not in self._spots:
            return False
        self._spots[spot_id].x = float(x)
        self._spots[spot_id].y = float(y)
        return True

    def set_laser_power(self, spot_id: int, power: float) -> bool:
        if spot_id not in self._spots:
            return False
        self._spots[spot_id].power = float(np.clip(power, 0.0, 1.0))
        return True

    def get_camera_frame(self) -> Optional[CameraFrame]:
        now = time.time()
        if now - self._last_frame_time < 1.0 / self.frame_rate:
            time.sleep(1.0 / self.frame_rate - (now - self._last_frame_time))

        # Generate synthetic image with current laser spots (reuse cached background)
        spot_positions = [(s.x, s.y) for s in self._spots.values() if s.active]
        image, _ = self._microscope.generate(
            self._simulated_robots,
            laser_spots=spot_positions if spot_positions else None,
            background=self._cached_background
        )

        self._last_frame_time = time.time()
        return CameraFrame(
            timestamp=self._last_frame_time,
            image=image,
            exposure_us=10000.0,
            gain_db=0.0
        )

    def get_spot_positions(self) -> List[LaserSpot]:
        return list(self._spots.values())

    def step_physics(self):
        """Advance physics simulation for trapped robots."""
        from simulation.optical_tweezer_sim import Trap

        active_traps = [Trap(s.x, s.y, s.power, s.active)
                        for s in self._spots.values()]

        for i, state in enumerate(self._robot_states):
            new_state = self._ot_sim.step(state, active_traps)
            self._robot_states[i] = new_state
            self._simulated_robots[i].x = new_state.x
            self._simulated_robots[i].y = new_state.y


# ------------------------------------------------------------------
# TODO: Real hardware implementations
# ------------------------------------------------------------------

class ThorlabsOTHardware(HardwareInterface):
    """
    Placeholder for Thorlabs optical tweezer hardware control.
    Implement using Thorlabs Kinesis SDK or similar.
    """

    def connect(self) -> bool:
        raise NotImplementedError(
            "Thorlabs hardware support not yet implemented. "
            "Please implement using your specific SDK."
        )

    def disconnect(self):
        pass

    def move_laser_spot(self, spot_id: int, x: float, y: float) -> bool:
        raise NotImplementedError

    def set_laser_power(self, spot_id: int, power: float) -> bool:
        raise NotImplementedError

    def get_camera_frame(self) -> Optional[CameraFrame]:
        raise NotImplementedError

    def get_spot_positions(self) -> List[LaserSpot]:
        raise NotImplementedError


class ElliotE3500Hardware(HardwareInterface):
    """
    Placeholder for Elliot Scientific E3500 OT system control.
    The actual system used in Ren et al. (MARSS 2022).
    """

    def connect(self) -> bool:
        raise NotImplementedError(
            "Elliot E3500 hardware support not yet implemented. "
            "Please implement using Elliot Scientific SDK or serial commands."
        )

    def disconnect(self):
        pass

    def move_laser_spot(self, spot_id: int, x: float, y: float) -> bool:
        raise NotImplementedError

    def set_laser_power(self, spot_id: int, power: float) -> bool:
        raise NotImplementedError

    def get_camera_frame(self) -> Optional[CameraFrame]:
        raise NotImplementedError

    def get_spot_positions(self) -> List[LaserSpot]:
        raise NotImplementedError


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_hardware(mode: str = "simulation",
                    **kwargs) -> HardwareInterface:
    """
    Factory function to create hardware interface.

    Args:
        mode: 'simulation', 'thorlabs', or 'elliot_e3500'
    """
    if mode == "simulation":
        return SimulationHardware(**kwargs)
    elif mode == "thorlabs":
        return ThorlabsOTHardware(**kwargs)
    elif mode == "elliot_e3500":
        return ElliotE3500Hardware(**kwargs)
    else:
        raise ValueError(f"Unknown hardware mode: {mode}")


if __name__ == "__main__":
    hw = create_hardware("simulation")
    hw.connect()

    frame = hw.get_camera_frame()
    print(f"Frame shape: {frame.image.shape}")
    cv2.imwrite("/tmp/sim_hw_frame.jpg", frame.image)

    hw.move_laser_spot(0, 200, 200)
    hw.move_laser_spot(1, 400, 300)
    spots = hw.get_spot_positions()
    print(f"Spot positions: {[(s.x, s.y) for s in spots[:2]]}")

    hw.disconnect()
    print("Simulation hardware test complete.")
