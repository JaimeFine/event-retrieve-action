# pyright: reportMissingImports=false
from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicSphere


class BaseIntruder:
    def __init__(self, name, position, color, radius=0.5):
        self.name = name
        self.initial_position = np.array(position, dtype=float)
        self.radius = radius

        self.prim = DynamicSphere(
            prim_path=f"/World/{name}",
            name=name,
            position=self.initial_position,
            radius=self.radius,
            color=np.array(color, dtype=float),
        )

    def apply_behavior(self):
        return

    def _on_state_set(self, velocity):
        return

    def set_state(self, position, velocity):
        position = np.array(position, dtype=float)
        velocity = np.array(velocity, dtype=float)
        self.prim.set_world_pose(position=position)
        self.prim.set_linear_velocity(velocity)
        self._on_state_set(velocity)

    def get_state(self):
        pos, _ = self.prim.get_world_pose()
        vel = self.prim.get_linear_velocity()
        return pos, vel

    def reset(self):
        self.prim.set_world_pose(position=self.initial_position)
        self.prim.set_linear_velocity(np.zeros(3))
        self.prim.set_angular_velocity(np.zeros(3))
        self._on_state_set(np.zeros(3))
