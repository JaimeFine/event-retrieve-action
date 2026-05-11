from __future__ import annotations

import numpy as np

from bruce_code.sim_shared.constants import bird

from .base import BaseIntruder


class BirdIntruder(BaseIntruder):
    def __init__(
        self,
        name,
        position,
        base_velocity=None,
        frequency=1.0,
        amplitude=0.6,
    ):
        super().__init__(name, position, color=bird, radius=0.25)
        self.base_velocity = np.array(base_velocity if base_velocity is not None else [-2.0, 0.0, 0.0], dtype=float)
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.step_count = 0
        self.dt = 0.05

    def _on_state_set(self, velocity):
        self.base_velocity = np.array(velocity, dtype=float)
        self.step_count = 0

    def apply_behavior(self):
        self.step_count += 1
        t = self.step_count * self.dt
        current_vel = np.array(self.base_velocity, dtype=float)
        current_vel[1] += np.sin(t * self.frequency) * self.amplitude
        self.prim.set_linear_velocity(current_vel)

    def reset(self):
        super().reset()
        self.step_count = 0
