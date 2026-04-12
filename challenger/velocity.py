import numpy as np
from macro import MIN_DST, MAX_DST

class EncounterSampler:
    def __init__(self, rng):
        self.rng = rng

    def sample_encounter(self, ego_pos, ego_vel, difficulty):
        """
        Returns: spawn_pos, velocity
        Enforces:
        - Controlled TTC
        - Controlled minimum distance
        - Controlled geometry
        """

        # 1. Difficulty Mapping
        ttc = np.interp(difficulty, [0, 1], [5.0, 1.0])
        d_min = np.interp(difficulty, [0, 1], [3.0, 0.3])
        speed_min, speed_max = 2.0, 6.0

        # 2. Sample Approach Direction
        theta = self.rng.uniform(0, np.pi)
 
        # Ego forward direction
        ego_dir = ego_vel if np.linalg.norm(ego_vel) > 1e-3 \
            else np.array([1.0, 0.0, 0.0])
        ego_dir /= np.linalg.norm(ego_dir)

        # Build orthonormal basis
        if abs(ego_dir[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        v1 = np.cross(ego_dir, ref)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(ego_dir, v1)

        approach_dir = np.cos(theta) * ego_dir + np.sin(theta) * v1

        # 3. Spawn Position
        spawn_dist = self.rng.uniform(MIN_DST, MAX_DST)
        base_spawn = ego_pos + approach_dir * spawn_dist

        if abs(approach_dir[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        v1 = np.cross(approach_dir, ref)
        v1 /= np.linalg.norm(v1)

        v2 = np.cross(approach_dir, v1)

        noise_radius = np.interp(difficulty, [0, 1], [0.5, 4.0])

        r1 = self.rng.uniform(0, noise_radius)
        r2 = self.rng.uniform(0, noise_radius * 0.3)

        lateral_noise = r1 * v1 + r2 * v2

        spawn_pos = base_spawn + lateral_noise

        # 4. Near-Miss Target
        offset_dir = v2
        target_point = ego_pos + offset_dir * d_min

        # 5. Velocity Solve
        velocity = (target_point - spawn_pos) / ttc

        # Clip velocity
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            velocity = approach_dir * speed_min
        else:
            velocity = velocity / speed * np.clip(speed, speed_min, speed_max)

        return spawn_pos, velocity