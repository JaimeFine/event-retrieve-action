import numpy as np
from .base import BaseIntruder

class StaticObstacle(BaseIntruder):
    def __init__(self, name, position):
        super().__init__(name, position, color=np.array([0.5, 0.5, 0.5]))

    def apply_behavior(self):
        # Environment constraints (buildings, poles) do not move
        self.prim.set_linear_velocity(np.zeros(3))