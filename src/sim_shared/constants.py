from __future__ import annotations

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_phys = 0.1
lambda_perf = 1.0

SAFETY_THRESHOLD = 1.0

BATCH_SIZE = 16
EPOCHS = 10
seeds = 25

detection_threshold = 3.0

ego = np.array([0.0, 0.0, 1.5])
bird = np.array([0.0, 1.0, 0.0])
drone = np.array([1.0, 0.0, 0.0])
static = np.array([0.5, 0.5, 0.5])

total_epochs = 100

NUM_INTRUDERS = 25
MAX_DST = 5.0
MIN_DST = 3.0
