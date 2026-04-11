from .base import BaseIntruder
import numpy as np

class BirdIntruder(BaseIntruder):   # GREEN
    def __init__(
        self, name, position, base_velocity=None,
        frequency=2.0, amplitude=2.0
    ):
        super().__init__(
            name, position, color=np.array([0.0, 1.0, 0.0]), radius=0.15
        )
        self.base_velocity = base_velocity if base_velocity is not None \
            else np.array([-2.0, 0.0, 0.0])
        self.frequency = frequency
        self.amplitude = amplitude

        self.step_count = 0
        self.dt = 0.05

    def apply_behavior(self):
        # Erratic, sinusoidal flight pattern
        self.step_count += 1
        
        t = self.step_count * self.dt
        y_vel = np.sin(t * self.frequency) * self.amplitude

        current_vel = np.copy(self.base_velocity)
        current_vel[1] = y_vel

        self.prim.set_linear_velocity(current_vel)

    def reset(self):
        super().reset()

        self.step_count = 0