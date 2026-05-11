from __future__ import annotations

import torch
import torch.nn as nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:
    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, observation_space, features_dim: int):
            super().__init__()
            self.observation_space = observation_space
            self.features_dim = features_dim


PPO_ARCHITECTURE_NAME = "threat_aware_transformer"
PPO_MAX_INTRUDERS = 5
PPO_GLOBAL_DIM = 7
PPO_INTRUDER_DIM = 8
PPO_OBSERVATION_DIM = PPO_GLOBAL_DIM + PPO_MAX_INTRUDERS * PPO_INTRUDER_DIM


class ThreatAwareTransformerExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        # We keep the feature width and embedding width both at 128 so the
        # extractor has enough capacity to model multi-intruder interactions
        # without making the PPO head disproportionately larger than BC/ERA.
        self.global_proj = nn.Sequential(
            nn.Linear(PPO_GLOBAL_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.intruder_proj = nn.Sequential(
            nn.Linear(PPO_INTRUDER_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        # One learnable slot per token: 1 global token + fixed intruder slots.
        # This lets the transformer distinguish the global state token from the
        # threat tokens even though all inputs are packed into one sequence.
        self.slot_embedding = nn.Parameter(torch.zeros(1, PPO_MAX_INTRUDERS + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Two encoder layers and four attention heads were chosen as a moderate
        # architecture: deep enough to model interactions among threats, but
        # still lightweight enough for repeated comparison runs on Isaac Sim
        # hardware without turning PPO into the dominant runtime bottleneck.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Sequential(
            nn.Linear(embed_dim * 2, features_dim),
            nn.SiLU(),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        global_obs = observations[:, :PPO_GLOBAL_DIM]
        intruders = observations[:, PPO_GLOBAL_DIM:].view(-1, PPO_MAX_INTRUDERS, PPO_INTRUDER_DIM)

        global_token = self.global_proj(global_obs).unsqueeze(1)
        intruder_tokens = self.intruder_proj(intruders)
        tokens = torch.cat([global_token, intruder_tokens], dim=1) + self.slot_embedding

        rel_pos = intruders[:, :, 1:4]
        # Zero rows are padding rows inserted to keep a fixed observation size.
        # We mask them so the transformer only attends to real threats.
        valid_intruders = torch.norm(rel_pos, dim=-1) > 1e-6
        padding_mask = torch.cat(
            [
                torch.zeros((observations.shape[0], 1), dtype=torch.bool, device=observations.device),
                ~valid_intruders,
            ],
            dim=1,
        )

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        cls_token = encoded[:, 0]
        masked_intruders = encoded[:, 1:] * valid_intruders.unsqueeze(-1)
        pooled_intruders = masked_intruders.sum(dim=1) / valid_intruders.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.pool(torch.cat([cls_token, pooled_intruders], dim=-1))


def get_ppo_policy_kwargs() -> dict:
    return {
        "features_extractor_class": ThreatAwareTransformerExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 2,
        },
        # Use a slightly larger actor/critic head than the extractor output so
        # PPO can separate policy/value features after the shared transformer,
        # while remaining small enough to keep training time comparable.
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
        "activation_fn": torch.nn.SiLU,
        "ortho_init": False,
    }


__all__ = [
    "PPO_ARCHITECTURE_NAME",
    "PPO_GLOBAL_DIM",
    "PPO_INTRUDER_DIM",
    "PPO_MAX_INTRUDERS",
    "PPO_OBSERVATION_DIM",
    "ThreatAwareTransformerExtractor",
    "get_ppo_policy_kwargs",
]
