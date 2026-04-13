# pyright: reportMissingImports=false
import numpy as np
from isaacsim.core.api.objects import DynamicSphere

class BaseIntruder:
    def __init__(self, name, position, color, radius=0.5):
        self.name = name
        self.initial_position = np.array(position)
        self.radius = radius

        self.prim = DynamicSphere(
            prim_path=f"/World/{name}",
            name=name,
            position=self.initial_position,
            radius=self.radius,
            color=np.array(color)
        )

    def apply_behavior(self):
        """To be overridden by child classes"""
        pass

    def set_state(self, position, velocity):
        """Teleports the intruder and applies a new velocity vector."""
        self.prim.set_world_pose(position=position)
        self.prim.set_linear_velocity(velocity)

    def get_state(self):
        pos, _ = self.prim.get_world_pose()
        vel = self.prim.get_linear_velocity()

        return pos, vel
    
    def reset(self):
        """
        For seeded scenarios: Zeroes out physics forces and returns the
        intruder to its exact spawn point.
        """
        self.prim.set_world_pose(position=self.initial_position)
        self.prim.set_linear_velocity(np.zeros(3))
        self.prim.set_angular_velocity(np.zeros(3))