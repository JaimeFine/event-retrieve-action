from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - fallback for older stacks
    gym = None

from .architectures import (
    PPO_ARCHITECTURE_NAME,
    PPO_GLOBAL_DIM,
    PPO_INTRUDER_DIM,
    PPO_MAX_INTRUDERS,
    PPO_OBSERVATION_DIM,
    get_ppo_policy_kwargs,
)
from .callbacks import PPOTrainingRecorder
from .reporting import save_training_curves


def get_default_ppo_observation_dim() -> int:
    return PPO_OBSERVATION_DIM


def event_list_to_observation(
    event_list: torch.Tensor | None,
    feature_dim: int = PPO_OBSERVATION_DIM,
) -> np.ndarray:
    if event_list is None or event_list.numel() == 0:
        return np.zeros(feature_dim, dtype=np.float32)

    if not isinstance(event_list, torch.Tensor):
        event_list = torch.tensor(event_list, dtype=torch.float32)

    events = event_list.detach().cpu().float()
    if events.dim() == 1:
        events = events.unsqueeze(0)

    if events.shape[-1] < 13:
        obs = torch.zeros(feature_dim, dtype=torch.float32)
        flat = events.reshape(-1)
        obs[: min(feature_dim, flat.numel())] = flat[: min(feature_dim, flat.numel())]
        return obs.numpy().astype(np.float32)

    rel_pos = events[:, 1:4]
    dists = torch.norm(rel_pos, dim=1)
    order = torch.argsort(dists)
    # Keep only the nearest threats so the PPO observation size stays fixed.
    # We use the top-k nearest intruders instead of all detected objects because
    # the action-relevant risk is dominated by nearby conflicts, while a fixed
    # input width is required by stable-baselines3 policies.
    top = order[:PPO_MAX_INTRUDERS]

    ego_vel = events[0, 7:10]
    rel_goal = events[0, 10:13]

    obs = torch.zeros(PPO_OBSERVATION_DIM, dtype=torch.float32)
    obs[0:3] = ego_vel
    obs[3:6] = rel_goal
    # Normalize the active-intruder count to [0, 1] so this scalar stays on the
    # same rough scale as the other observation channels.
    obs[6] = float(min(events.shape[0], PPO_MAX_INTRUDERS)) / float(PPO_MAX_INTRUDERS)

    cursor = PPO_GLOBAL_DIM
    for idx in top:
        event = events[int(idx.item())]
        dist = torch.norm(event[1:4])

        block = torch.zeros(PPO_INTRUDER_DIM, dtype=torch.float32)
        block[0] = event[0]
        block[1:4] = event[1:4]
        block[4:7] = event[4:7]
        block[7] = dist
        obs[cursor: cursor + PPO_INTRUDER_DIM] = block
        cursor += PPO_INTRUDER_DIM

    if feature_dim != PPO_OBSERVATION_DIM:
        if feature_dim < PPO_OBSERVATION_DIM:
            obs = obs[:feature_dim]
        else:
            padded = torch.zeros(feature_dim, dtype=obs.dtype)
            padded[:PPO_OBSERVATION_DIM] = obs
            obs = padded

    return obs.numpy().astype(np.float32)


class OfflineImitationEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        experiences: Iterable,
        observation_dim: int = PPO_OBSERVATION_DIM,
        action_dim: int = 3,
        max_speed: float = 5.0,
    ):
        if gym is None:
            raise ImportError("gymnasium is required for OfflineImitationEnv")
        spaces = gym.spaces

        self.experiences = list(experiences)
        if len(self.experiences) == 0:
            raise ValueError("experiences cannot be empty")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_speed = max_speed

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.max_speed,
            high=self.max_speed,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self._current_target_action = np.zeros(self.action_dim, dtype=np.float32)

    def _sample_experience(self):
        idx = int(self._rng.integers(0, len(self.experiences)))
        return self.experiences[idx]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        exp = self._sample_experience()
        obs = event_list_to_observation(exp.event_list, feature_dim=self.observation_dim)
        self._current_target_action = exp.action.detach().cpu().numpy().astype(np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        target = self._current_target_action.reshape(-1)
        reward = -float(np.mean((action - target) ** 2))

        exp = self._sample_experience()
        obs = event_list_to_observation(exp.event_list, feature_dim=self.observation_dim)
        self._current_target_action = exp.action.detach().cpu().numpy().astype(np.float32)

        terminated = True
        truncated = False
        info = {"target_action": target}
        return obs, reward, terminated, truncated, info


def build_ppo_model(env, policy: str | None = None, **kwargs):
    sb3 = importlib.import_module("stable_baselines3")

    policy_kwargs = dict(kwargs.pop("policy_kwargs", {}))
    for key, value in get_ppo_policy_kwargs().items():
        policy_kwargs.setdefault(key, value)
    kwargs["policy_kwargs"] = policy_kwargs

    return sb3.PPO(policy or "MlpPolicy", env, **kwargs)


def train_ppo_model(
    env,
    total_timesteps: int = 200_000,
    training_artifact_dir: str | None = None,
    run_name: str = "ppo",
    **kwargs,
):
    # The 200k-step default is a conservative standalone upper bound rather than
    # a paper claim that convergence always happens by 200k steps. In the actual
    # comparison pipeline, train_ppo_from_experiences() usually overrides this
    # with a budget tied to dataset size and epochs so PPO sees a budget that is
    # easier to compare against BC. We keep 200k here so direct calls to this
    # helper do not terminate after only a few updates on larger datasets.
    model = build_ppo_model(env, **kwargs)
    recorder = PPOTrainingRecorder()
    model.learn(total_timesteps=total_timesteps, callback=recorder)
    training_history = recorder.export()
    training_artifacts = None
    if training_artifact_dir:
        training_artifacts = save_training_curves(training_history, training_artifact_dir, run_name=run_name)
    return model, training_history, training_artifacts


class StableBaselinesPolicyAdapter:
    def __init__(
        self,
        model,
        observation_dim: int = PPO_OBSERVATION_DIM,
        training_history: dict | None = None,
        training_artifacts: dict | None = None,
    ):
        self.model = model
        self.observation_dim = observation_dim
        self.architecture = PPO_ARCHITECTURE_NAME
        self.training_history = training_history or {}
        self.training_artifacts = training_artifacts or {}

    def reset(self) -> None:
        return

    @torch.no_grad()
    def predict(self, event_list: torch.Tensor | None) -> torch.Tensor:
        obs = event_list_to_observation(event_list, feature_dim=self.observation_dim)
        action, _ = self.model.predict(obs, deterministic=True)
        return torch.tensor(action, dtype=torch.float32)


def save_stable_baselines_policy_adapter(policy: StableBaselinesPolicyAdapter, path: str) -> None:
    policy.model.save(path)


def load_stable_baselines_policy_adapter(
    path: str,
    observation_dim: int = PPO_OBSERVATION_DIM,
    device: str = "cpu",
) -> StableBaselinesPolicyAdapter:
    sb3 = importlib.import_module("stable_baselines3")
    model = sb3.PPO.load(path, device=device)
    return StableBaselinesPolicyAdapter(
        model=model,
        observation_dim=observation_dim,
    )


def train_ppo_from_experiences(
    experiences,
    total_timesteps: int | None = None,
    learning_rate: float = 3e-4,
    observation_dim: int = PPO_OBSERVATION_DIM,
    action_dim: int = 3,
    max_speed: float = 5.0,
    device: str = "cpu",
    epochs: int = 10,
    batch_size: int = 16,
    seed: int = 25,
    training_artifact_dir: str | None = None,
    run_name: str = "ppo",
):
    env = OfflineImitationEnv(
        experiences=experiences,
        observation_dim=observation_dim,
        action_dim=action_dim,
        max_speed=max_speed,
    )
    if total_timesteps is None or total_timesteps <= 0:
        # For the main comparison script we derive the PPO budget from the same
        # epochs parameter used by BC: len(dataset) * epochs. This does not make
        # the training paradigms identical, but it does keep the default budget
        # roughly proportional to the amount of expert data exposed during
        # offline imitation.
        total_timesteps = max(len(experiences) * max(1, epochs), 2048)

    # PPO needs multiple rollout steps per update. We cap n_steps to stay within
    # a stable small-run regime for this offline-imitation setting, while still
    # scaling it with batch size so updates do not become too noisy when the
    # batch is increased for comparison runs.
    n_steps = max(128, min(512, batch_size * 8))

    model, training_history, training_artifacts = train_ppo_model(
        env,
        total_timesteps=total_timesteps,
        training_artifact_dir=training_artifact_dir,
        run_name=run_name,
        learning_rate=learning_rate,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        # These are standard PPO defaults kept close to the stable-baselines3
        # reference settings so the comparison does not depend on aggressive
        # retuning for our custom threat-aware encoder.
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=seed,
    )
    return StableBaselinesPolicyAdapter(
        model=model,
        observation_dim=observation_dim,
        training_history=training_history,
        training_artifacts=training_artifacts,
    )
