import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reinforcement Learning Parameters:
lambda_phys = 0.1
lambda_perf = 1.0

# Running Meta-Parameters:
SAFETY_THRESHOLD = 1.5  # Meters
COLLISION_CRITICAL = 0.5    # Meters

# Random Initialization Seeds:
seeds = 42

# The Detection Threshold:
detection_threshold = 5.0  # Meters

# Colors:
ego = np.array([0.0, 0.0, 1.5]) # Blue
