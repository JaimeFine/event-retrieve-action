from __future__ import annotations

import numpy as np

from bruce_code.sim_shared.constants import MAX_DST, MIN_DST
from bruce_code.sim_shared.difficulty import DifficultyProfile


class EncounterSampler:
    def __init__(self, rng):
        self.rng = rng

    def sample_encounter(self, ego_pos, ego_vel, profile: DifficultyProfile):
        ttc = self.rng.uniform(*profile.ttc_range)
        d_min = self.rng.uniform(*profile.min_distance_range)
        speed_min, speed_max = profile.speed_range

        theta = self.rng.uniform(0.0, np.pi)

        ego_dir = ego_vel if np.linalg.norm(ego_vel) > 1e-3 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ego_dir = ego_dir / (np.linalg.norm(ego_dir) + 1e-6)

        ref = np.array([1.0, 0.0, 0.0]) if abs(ego_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v1 = np.cross(ego_dir, ref)
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = np.cross(ego_dir, v1)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)

        approach_dir = np.cos(theta) * ego_dir + np.sin(theta) * v1
        spawn_min = max(MIN_DST, profile.spawn_distance_range[0])
        spawn_max = min(MAX_DST + 2.5, profile.spawn_distance_range[1])
        spawn_dist = self.rng.uniform(spawn_min, max(spawn_min + 1e-3, spawn_max))
        base_spawn = ego_pos + approach_dir * spawn_dist

        noise_min, noise_max = profile.lateral_noise_range
        lateral_noise = self.rng.uniform(noise_min, noise_max) * v1
        vertical_noise = self.rng.uniform(0.0, noise_max * 0.35) * v2
        spawn_pos = base_spawn + lateral_noise + vertical_noise

        future_ego_pos = ego_pos + ego_vel * ttc
        target_point = future_ego_pos + v2 * d_min
        velocity = (target_point - spawn_pos) / max(ttc, 1e-3)

        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            velocity = approach_dir * speed_min
        else:
            velocity = velocity / speed * np.clip(speed, speed_min, speed_max)

        return spawn_pos.astype(np.float32), velocity.astype(np.float32)
