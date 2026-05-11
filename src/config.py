from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    input_dim: int = 13
    latent_dim: int = 128
    hidden_dim: int = 256
    action_dim: int = 3
    use_inverse_norm_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalConfig:
    k: int = 5
    temperature: float = 0.1
    similarity_threshold: float = 0.8
    contraction_margin: float = 0.99
    use_bayesian_clustering: bool = True
    use_physics_regularizer: bool = True
    fallback_to_nearest: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    metric_weight: float = 0.5
    imitation_weight: float = 1.0
    physics_weight: float = 0.5
    dt: float = 0.05
    device: str = "cpu"
    checkpoint_path: str = "agent_pretrained.pt"

    def to_dict(self) -> dict:
        return asdict(self)
