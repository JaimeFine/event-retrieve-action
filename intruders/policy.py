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
        return torch.tensor(
            np.concatenate([rel_pos, intruder_vel]),
            dtype=torch.float32,
            device=device
        )
    
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
        self.log_probs.extend(log_probs)
        self.rewards.extend(rewards)

    def update(self):
        if len(self.rewards) == 0:
            return 0.0

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss += -log_prob * G   # POLICY GRADIENT

        loss = loss / len(returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffer
        self.log_probs = []
        self.rewards = []

        return loss.item()
    
def apply_multiagent_intruder_behavior(controller, ego, intruders):
    ego_pos, _ = ego.get_world_pose()

    log_probs = []
    intruder_positions = []

    # ---- ACTION PHASE ----
    for intr in intruders:
        pos, vel = intr.get_state()
        state = controller.get_state(ego_pos, pos, vel)

        action, log_prob = controller.select_action(state)

        # Apply velocity update
        current_vel = intr.prim.get_linear_velocity()
        new_vel = current_vel + action.detach().cpu().numpy()

        # Speed clamp
        speed = np.linalg.norm(new_vel)
        if speed > 6.0:
            new_vel = new_vel / speed * 6.0

        intr.prim.set_linear_velocity(new_vel)

        log_probs.append(log_prob)
        intruder_positions.append(pos)

    # ---- REWARD PHASE ----
    rewards = controller.compute_multiagent_reward(
        ego_pos,
        intruder_positions
    )

    # Store trajectories
    controller.store(log_probs, rewards)
