from .base import BaseIntruder
import numpy as np

class DroneIntruder(BaseIntruder):  # RED
    def __init__(self, name, position, velocity=None):
        super().__init__(name, position, color=np.array([1.0, 0.0, 0.0]))
        self.velocity = velocity if velocity is not None \
            else np.array([-3.0, 0.0, 0.0])

    def apply_behavior(self):
        self.prim.set_linear_velocity(self.velocity)