from __future__ import annotations

import numpy as np

from bruce_code.sim_shared.difficulty import DifficultyProfile

from .velocity import EncounterSampler


class RuleBasedAdversarialSpawner:
    def __init__(self, rng):
        self.rng = rng
        self.sampler = EncounterSampler(rng)

    def spawn_event(self, ego_pos, ego_vel, profile: DifficultyProfile):
        event_types = list(profile.event_weights.keys())
        weights = np.asarray([profile.event_weights[item] for item in event_types], dtype=np.float32)
        weights = weights / weights.sum()
        event_type = self.rng.choice(event_types, p=weights)

        if event_type == "collision_course":
            return self._collision_case(ego_pos, ego_vel, profile)
        if event_type == "near_miss":
            return self._near_miss_case(ego_pos, ego_vel, profile)
        if event_type == "crossing":
            return self._crossing_case(ego_pos, ego_vel, profile)
        return self._multi_intruder_case(ego_pos, ego_vel, profile)

    def _collision_case(self, ego_pos, ego_vel, profile: DifficultyProfile):
        spawn_pos, velocity = self.sampler.sample_encounter(ego_pos, ego_vel, profile)
        return [(spawn_pos, velocity)]

    def _near_miss_case(self, ego_pos, ego_vel, profile: DifficultyProfile):
        softened = DifficultyProfile(
            **{
                **profile.__dict__,
                "min_distance_range": (
                    profile.min_distance_range[0] * 1.2,
                    profile.min_distance_range[1] * 1.4,
                ),
            }
        )
        spawn_pos, velocity = self.sampler.sample_encounter(ego_pos, ego_vel, softened)
        return [(spawn_pos, velocity)]

    def _crossing_case(self, ego_pos, ego_vel, profile: DifficultyProfile):
        ego_dir = ego_vel if np.linalg.norm(ego_vel) > 1e-3 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ego_dir = ego_dir / (np.linalg.norm(ego_dir) + 1e-6)

        perp_dir = np.array([-ego_dir[1], ego_dir[0], 0.0], dtype=np.float32)
        if np.linalg.norm(perp_dir) < 1e-3:
            perp_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-6)

        spawn_radius = self.rng.uniform(*profile.spawn_distance_range)
        spawn_pos = ego_pos + perp_dir * spawn_radius
        speed = self.rng.uniform(*profile.speed_range)
        velocity = -perp_dir * speed
        return [(spawn_pos.astype(np.float32), velocity.astype(np.float32))]

    def _multi_intruder_case(self, ego_pos, ego_vel, profile: DifficultyProfile):
        intruder_count = 2 if profile.name in {"easy", "medium"} else 3
        cases = []
        for idx in range(intruder_count):
            spawn_pos, velocity = self.sampler.sample_encounter(ego_pos, ego_vel, profile)
            velocity = velocity + np.array([0.0, (idx - 1) * 1.25, 0.0], dtype=np.float32)
            cases.append((spawn_pos.astype(np.float32), velocity.astype(np.float32)))
        return cases
