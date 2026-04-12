import numpy as np

class CurriculumScheduler:
    def __init__(self, total_steps):
        self.total_steps = total_steps

        # Performance tracker
        self.recent_rewards = []
        self.window_size = 50   # smoothing window

    def update_performance(self, reward):
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)

    def get_success_rate(self):
        if len(self.recent_rewards) == 0:
            return 0.0
        
        rewards = np.array(self.recent_rewards)
        
        # Define success: reward > 0
        success = (rewards > 0.5).astype(float)
        return success.mean()

    def get_difficulty(self, step):
        # --- 1. Base curriculum (time-based) ---
        progress = step / self.total_steps
        base_difficulty = 1 / (1 + np.exp(-10 * (progress - 0.5)))  # sigmoid

        # --- 2. Performance adaptation ---
        success_rate = self.get_success_rate()

        adaptive_term = 0.3 * success_rate

        # Smooth ramp
        difficulty = np.clip(
            0.5 + base_difficulty + adaptive_term,
            0.0, 1.0
        )

        return float(difficulty)