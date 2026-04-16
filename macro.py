import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reinforcement Learning Parameters:
lambda_phys = 0.1
lambda_perf = 1.0

# Running Meta-Parameters:
SAFETY_THRESHOLD = 1.0  # Meters

# Training Parameters:
BATCH_SIZE = 16
EPOCHS = 10

# Random Initialization Seeds:
seeds = 25

# The Detection Threshold:
detection_threshold = 3.0  # Meters

# Colors:
ego = np.array([0.0, 0.0, 1.5]) # Blue
bird = np.array([0.0, 1.0, 0.0]) # Green
drone = np.array([1.0, 0.0, 0.0]) # Red
static = np.array([0.5, 0.5, 0.5]) # Grey

# Total Epochs for the main loop:
total_epochs = 100

# Number of Intruders in the Environment:
NUM_INTRUDERS = 25

# Intruders Spawning Parameters:
MAX_DST = 5.0
MIN_DST = 3.0