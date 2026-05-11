from __future__ import annotations

from bruce_code.methods.era.external import build_external_era_agent
from bruce_code.sim_shared.constants import device
from bruce_code.sim_shared.environment import SharedIsaacEnvironment


class EraIsaacEnvironment(SharedIsaacEnvironment):
    def __init__(
        self,
        seed: int,
        difficulty_mode: str = "curriculum",
        difficulty_level: str = "medium",
    ):
        agent = build_external_era_agent(latent_dim=128).to(device)
        super().__init__(
            agent=agent,
            seed=seed,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
        )
