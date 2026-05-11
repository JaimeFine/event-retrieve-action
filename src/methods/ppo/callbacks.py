from __future__ import annotations

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:
    class BaseCallback:
        def __init__(self):
            self.locals = {}
            self.logger = type("_DummyLogger", (), {"name_to_value": {}})()
            self.num_timesteps = 0


class PPOTrainingRecorder(BaseCallback):
    def __init__(self):
        super().__init__()
        self.reward_curve: list[dict[str, float]] = []
        self.rollout_curve: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            try:
                reward_value = float(rewards.mean().item())
            except Exception:
                reward_value = float(rewards)
            self.reward_curve.append(
                {
                    "timesteps": float(self.num_timesteps),
                    "reward": reward_value,
                }
            )
        return True

    def _on_rollout_end(self) -> None:
        logger_values = getattr(self.logger, "name_to_value", {})
        tracked = {
            "timesteps": float(self.num_timesteps),
        }
        for key in (
            "rollout/ep_rew_mean",
            "train/loss",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/explained_variance",
        ):
            value = logger_values.get(key)
            if value is not None:
                tracked[key] = float(value)
        self.rollout_curve.append(tracked)

    def export(self) -> dict:
        return {
            "reward_curve": list(self.reward_curve),
            "convergence_curve": list(self.rollout_curve),
        }
