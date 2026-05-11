from __future__ import annotations

import numpy as np

from bruce_code.sim_shared.constants import static

from .base import BaseIntruder


class StaticObstacle(BaseIntruder):
    def __init__(self, name, position):
        super().__init__(name, position, color=static, radius=1.5)

    def apply_behavior(self):
        self.prim.set_linear_velocity(np.zeros(3))
