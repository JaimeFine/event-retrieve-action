import torch.nn as nn
from macro import device, SAFETY_THRESHOLD
import torch
import numpy as np

class IntruderPolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean(h)
        std = torch.exp(self.log_std)
        return mean, std
    
class MultiAgentIntruderController:
    def __init__(self, lr=1e-3, gamma=0.99):
        self.policy = IntruderPolicy().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # Trajectory buffers
        self.log_probs = []
        self.rewards = []

    def get_state(self, ego_pos, intruder_pos, intruder_vel):
        rel_pos = intruder_pos - ego_pos
        return np.concatenate([rel_pos, intruder_vel]).astype(np.float32)
    
    def select_action(self, state):
        mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action, log_prob
    
    def compute_multiagent_reward(self, ego_pos, intruder_positions):
        rewards = []

        dists = [np.linalg.norm(p - ego_pos) for p in intruder_positions]
        min_dist = min(dists)

        for i, pos in enumerate(intruder_positions):
            dist = dists[i]

            # --- COLLISION REWARD (PRIMARY OBJECTIVE) ---
            if dist < 0.5:
                r = 50.0

            # --- DANGEROUS ZONE ---
            elif dist < SAFETY_THRESHOLD:
                r = 10.0 / (dist + 1e-6)

            # --- APPROACH INCENTIVE ---
            else:
                r = -0.05 * dist

            # --- MULTI-AGENT COORDINATION BONUS ---
            # Encourage swarm compression near ego
            r += 5.0 / (min_dist + 1e-6)

            rewards.append(r)

        return rewards

    def store(self, log_probs, rewards):
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)

    def update(self):
        if len(self.rewards) == 0:
            return 0.0

        rewards = torch.stack(self.rewards)
        log_probs = torch.stack(self.log_probs) 

        # Compute discounted returns
        returns = torch.zeros_like(rewards)
        G = torch.zeros(rewards.size(1), device=device)

        for t in reversed(range(rewards.size(0))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

        return loss.item()
    
def apply_multiagent_intruder_behavior(controller, ego, intruders):
    # ---- GET EGO STATE ----
    ego_pos_np, _ = ego.get_world_pose()
    ego_pos = torch.tensor(ego_pos_np, dtype=torch.float32, device=device)

    # ---- COLLECT ALL INTRUDER STATES (CPU → GPU ONCE) ----
    positions = []
    velocities = []
    current_vels = []

    for intr in intruders:
        pos, vel = intr.get_state()
        positions.append(pos)
        velocities.append(vel)
        current_vels.append(intr.prim.get_linear_velocity())

    positions = torch.from_numpy(np.stack(positions)).float().to(device)
    velocities = torch.from_numpy(np.stack(velocities)).float().to(device)
    current_vels = torch.from_numpy(np.stack(current_vels)).float().to(device)

    # ---- BUILD STATES (GPU) ----
    rel_pos = positions - ego_pos  # broadcast
    states = torch.cat([rel_pos, velocities], dim=1)  # [N, 6]

    # ---- POLICY FORWARD (BATCHED) ----
    means, stds = controller.policy(states)
    dist = torch.distributions.Normal(means, stds)

    actions = dist.sample()
    log_probs = dist.log_prob(actions).sum(dim=1)  # [N]

    # ---- APPLY ACTIONS ----
    new_vels = current_vels + actions

    # Speed clamp (vectorized)
    speeds = torch.norm(new_vels, dim=1, keepdim=True)  # [N,1]
    scale = torch.clamp(6.0 / (speeds + 1e-6), max=1.0)  # [N,1]
    new_vels = new_vels * scale  # broadcasts to [N,3]

    # ---- WRITE BACK TO SIM (ONLY CPU CONVERSION HERE) ----
    new_vels_np = new_vels.detach().cpu().numpy()

    for i, intr in enumerate(intruders):
        intr.prim.set_linear_velocity(new_vels_np[i])

    # ---- REWARD (FULL GPU) ----
    dists = torch.norm(positions - ego_pos, dim=1)
    min_dist = torch.min(dists)

    rewards = torch.where(
        dists < 0.5,
        torch.tensor(50.0, device=device),
        torch.where(
            dists < SAFETY_THRESHOLD,
            10.0 / (dists + 1e-6),
            -0.05 * dists
        )
    )

    rewards += 5.0 / (min_dist + 1e-6)

    # ---- STORE ----
    controller.store(log_probs, rewards)
