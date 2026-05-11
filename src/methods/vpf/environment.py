from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.prims import RigidPrim

from bruce_code.sim_shared.constants import ego
from bruce_code.sim_shared.environment import SharedIsaacEnvironment
from bruce_code.sim_shared.intruders import BirdIntruder, DroneIntruder, StaticObstacle


class _NoOpMemory:
    def penalize_by_indices(self, index, factor=1.0):
        return None


class _PlaceholderAgent:
    def __init__(self):
        self.memory = _NoOpMemory()

    def select_action(self, event_list, k=5):
        raise RuntimeError("Adapter not installed yet.")


class VpfIsaacEnvironment(SharedIsaacEnvironment):
    """
    Shared evaluation environment variant for VPF-only benchmarking.

    The only intentional difference is that it skips add_default_ground_plane()
    to avoid Nucleus asset resolution hangs on headless servers.
    """

    def __init__(
        self,
        seed: int,
        difficulty_mode: str = "curriculum",
        difficulty_level: str = "medium",
    ):
        super().__init__(
            agent=_PlaceholderAgent(),
            seed=seed,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
        )

    def setup_environment(self):
        self.ego = DynamicSphere(
            prim_path="/World/ego_drone",
            name="ego_drone",
            position=np.array([0.0, 0.0, 1.5]),
            radius=0.25,
            color=ego,
        )
        self.world.scene.add(self.ego)

        self.intruders = []
        hidden_pos = [0.0, 0.0, -100.0]

        for i in range(self.num_intruders):
            self.intruders.append(DroneIntruder(f"pool_drone_{i}", hidden_pos))
            self.intruders.append(BirdIntruder(f"pool_bird_{i}", hidden_pos))
            self.intruders.append(StaticObstacle(f"pool_static_{i}", hidden_pos))

        for intruder in self.intruders:
            self.world.scene.add(intruder.prim)

        self.ego_view = RigidPrim(
            prim_paths_expr="/World/ego_drone",
            name="ego_view",
            track_contact_forces=False,
        )
        self.world.scene.add(self.ego_view)
