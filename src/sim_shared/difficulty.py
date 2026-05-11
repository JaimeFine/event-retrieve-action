from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    scalar: float
    active_intruders: int
    event_weights: dict[str, float]
    ttc_range: tuple[float, float]
    min_distance_range: tuple[float, float]
    speed_range: tuple[float, float]
    spawn_distance_range: tuple[float, float]
    lateral_noise_range: tuple[float, float]
    response_gain: float
    separation_gain: float
    bird_wiggle_amp: float
    max_intruder_speed: float


_PROFILES: dict[str, DifficultyProfile] = {
    "easy": DifficultyProfile(
        name="easy",
        scalar=0.15,
        # Start with a small active set so early comparison runs are not
        # dominated by immediate clutter. Six intruders is enough to generate
        # interaction without saturating the 3D scene from the first steps.
        active_intruders=6,
        event_weights={"collision_course": 3.0, "near_miss": 3.0, "crossing": 2.5, "multi_conflict": 1.0},
        ttc_range=(2.2, 3.0),
        min_distance_range=(2.2, 3.2),
        speed_range=(3.5, 5.0),
        spawn_distance_range=(4.5, 7.0),
        lateral_noise_range=(0.3, 1.4),
        response_gain=0.26,
        separation_gain=0.34,
        bird_wiggle_amp=0.14,
        max_intruder_speed=3.2,
    ),
    "medium": DifficultyProfile(
        name="medium",
        scalar=0.45,
        active_intruders=10,
        event_weights={"collision_course": 4.0, "near_miss": 2.5, "crossing": 2.0, "multi_conflict": 2.0},
        ttc_range=(1.4, 2.2),
        min_distance_range=(1.2, 2.0),
        speed_range=(4.5, 6.2),
        spawn_distance_range=(3.8, 6.0),
        lateral_noise_range=(0.2, 2.1),
        response_gain=0.32,
        separation_gain=0.3,
        bird_wiggle_amp=0.2,
        max_intruder_speed=4.0,
    ),
    "hard": DifficultyProfile(
        name="hard",
        scalar=0.72,
        active_intruders=16,
        event_weights={"collision_course": 5.0, "near_miss": 2.0, "crossing": 1.8, "multi_conflict": 3.2},
        ttc_range=(0.7, 1.5),
        min_distance_range=(0.5, 1.1),
        speed_range=(5.6, 7.2),
        spawn_distance_range=(3.2, 5.0),
        lateral_noise_range=(0.1, 2.8),
        response_gain=0.37,
        separation_gain=0.26,
        bird_wiggle_amp=0.27,
        max_intruder_speed=4.7,
    ),
    "extreme": DifficultyProfile(
        name="extreme",
        scalar=0.92,
        # The extreme profile stays below the full pool size on purpose: we want
        # dense, hard scenes, but still leave spare pooled objects so spawning
        # logic can refresh encounters without exhausting every intruder slot.
        active_intruders=22,
        event_weights={"collision_course": 5.5, "near_miss": 1.5, "crossing": 1.5, "multi_conflict": 4.0},
        ttc_range=(0.35, 0.9),
        min_distance_range=(0.2, 0.55),
        speed_range=(6.5, 8.0),
        spawn_distance_range=(2.6, 4.2),
        lateral_noise_range=(0.05, 3.2),
        response_gain=0.42,
        separation_gain=0.2,
        bird_wiggle_amp=0.34,
        max_intruder_speed=5.2,
    ),
}


def get_difficulty_profile(name: str) -> DifficultyProfile:
    key = name.strip().lower()
    if key not in _PROFILES:
        raise ValueError(f"Unsupported difficulty profile: {name!r}")
    return _PROFILES[key]


def get_difficulty_profile_names() -> list[str]:
    return list(_PROFILES.keys())


class RuleBasedDifficultyScheduler:
    def __init__(
        self,
        total_steps: int,
        mode: str = "curriculum",
        fixed_level: str = "medium",
    ):
        self.total_steps = max(1, int(total_steps))
        self.mode = mode.strip().lower()
        self.fixed_level = fixed_level.strip().lower()
        self.recent_rewards: list[float] = []
        self.window_size = 50

        if self.mode not in {"curriculum", "fixed"}:
            raise ValueError(f"Unsupported difficulty mode: {mode!r}")
        if self.fixed_level not in _PROFILES:
            raise ValueError(f"Unsupported fixed difficulty level: {fixed_level!r}")

    def update_performance(self, reward: float) -> None:
        self.recent_rewards.append(float(reward))
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)

    def get_success_rate(self) -> float:
        if not self.recent_rewards:
            return 0.0
        rewards = np.asarray(self.recent_rewards, dtype=np.float32)
        # A reward above 0.5 corresponds to "clearly safe or goal-progressing"
        # steps under the current reward design, so we use it as a coarse proxy
        # for recent success when adapting difficulty.
        success = (rewards > 0.5).astype(np.float32)
        return float(success.mean())

    def get_scalar(self, step: int) -> float:
        if self.mode == "fixed":
            return get_difficulty_profile(self.fixed_level).scalar

        progress = float(step) / float(self.total_steps)
        # Curriculum is driven mostly by episode progress (85%) and only mildly
        # by recent performance (20% additive term before clipping). This keeps
        # the schedule predictable for fair comparison, while still nudging the
        # environment upward when the agent is consistently handling the current
        # threat level.
        base_difficulty = 0.05 + 0.85 * progress
        adaptive_term = 0.2 * self.get_success_rate()
        difficulty = np.clip(base_difficulty + adaptive_term, 0.0, 1.0)
        return float(difficulty)

    def get_profile(self, step: int) -> DifficultyProfile:
        if self.mode == "fixed":
            return get_difficulty_profile(self.fixed_level)

        scalar = self.get_scalar(step)
        if scalar < 0.25:
            base = get_difficulty_profile("easy")
        elif scalar < 0.55:
            base = get_difficulty_profile("medium")
        elif scalar < 0.8:
            base = get_difficulty_profile("hard")
        else:
            base = get_difficulty_profile("extreme")
        return replace(base, scalar=scalar)
