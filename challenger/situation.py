from .velocity import EncounterSampler
import numpy as np

class AdversarialSpawner:
    def __init__(self, rng):
        self.rng = rng
        self.sampler = EncounterSampler(rng)

    def spawn_event(self, ego_pos, ego_vel, difficulty):
        event_type = self.rng.choice([
            "collision_course",
            "near_miss",
            "crossing",
            "multi_conflict"
        ])

        if event_type == "collision_course":
            return self._collision_case(ego_pos, ego_vel, difficulty)

        elif event_type == "near_miss":
            return self._near_miss_case(ego_pos, ego_vel, difficulty)

        elif event_type == "crossing":
            return self._crossing_case(ego_pos, ego_vel, difficulty)

        elif event_type == "multi_conflict":
            return self._multi_intruder_case(ego_pos, ego_vel, difficulty)
        
    # --- CASES ---

    def _collision_case(self, ego_pos, ego_vel, difficulty):
        spawn_pos, velocity = self.sampler.sample_encounter(
            ego_pos, ego_vel, difficulty
        )
        return [(spawn_pos, velocity)]

    def _near_miss_case(self, ego_pos, ego_vel, difficulty):
        spawn_pos, velocity = self.sampler.sample_encounter(
            ego_pos, ego_vel, difficulty * 0.7
        )
        return [(spawn_pos, velocity)]

    def _crossing_case(self, ego_pos, ego_vel, difficulty):
        # Force perpendicular crossing
        perp_dir = np.array([-ego_vel[1], ego_vel[0], 0.0])
        if np.linalg.norm(perp_dir) < 1e-3:
            perp_dir = np.array([0.0, 1.0, 0.0])

        perp_dir /= np.linalg.norm(perp_dir)

        spawn_pos = ego_pos + perp_dir * self.rng.uniform(5.0, 12.0)

        velocity = -perp_dir * self.rng.uniform(2.0, 5.0)

        return [(spawn_pos, velocity)]

    def _multi_intruder_case(self, ego_pos, ego_vel, difficulty):
        # Two conflicting intruders
        cases = []

        for sign in [-1, 1]:
            spawn_pos, velocity = self.sampler.sample_encounter(
                ego_pos, ego_vel, difficulty
            )

            # Force divergence
            velocity += sign * np.array([0.0, 1.5, 0.0])
            cases.append((spawn_pos, velocity))

        return cases