from __future__ import annotations

import numpy as np

from bruce_code.sim_shared.constants import drone

from .base import BaseIntruder


class DroneIntruder(BaseIntruder):
    def __init__(self, name, position, velocity=None):
        super().__init__(name, position, color=drone)
        self.velocity = np.array(velocity if velocity is not None else [-3.0, 0.0, 0.0], dtype=float)

    def _on_state_set(self, velocity):
        self.velocity = np.array(velocity, dtype=float)

    def apply_behavior(self):
        self.prim.set_linear_velocity(self.velocity)
