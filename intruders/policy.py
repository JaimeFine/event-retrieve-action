import torch.nn as nn
from macro import device, COLLISION_CRITICAL, SAFETY_THRESHOLD
import torch
import numpy as np
import torch.nn.functional as F

class IntruderPolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class MultiAgentIntruderController:
    def __init__(self, lr=1e-3):
        self.policy = IntruderPolicy().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def get_state(self, ego_pos, intruder_pos, intruder_vel):
        rel_pos = intruder_pos - ego_pos
        return torch.tensor(
            np.concatenate([rel_pos, intruder_vel]),
            dtype=torch.float32,
            device=device
        )
    
    def select_action(self, states):
        actions = []
        for s in states:
            a = self.policy(s)
            actions.append(a)
        return actions
    
    def compute_reward(self, ego_pos, intruder_position):
        rewards = []

        for pos in intruder_position:
            dist = np.linalg.norm(pos - ego_pos)
            if dist < COLLISION_CRITICAL:
                r = 10.0
            elif dist < SAFETY_THRESHOLD:
                r = 5.0 / (dist + 1e-6)
            else:
                r = -0.1 * dist
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    
    def train_step(self, states, actions, rewards):
        self.optimizer.zero_grad()
        loss = 0.0
        for s, a, r in zip(states, actions, rewards):
            pred = self.policy(s)
            # MSE as surrogate
            loss += F.mse_loss(pred, a.detach()) * (-r)
        loss = loss / len(states)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
def apply_multiagent_intruder_behavior(controller, ego, intruders):
    ego_pos, _ = ego.get_world_pose()

    states = []
    intruder_positions = []
    intruder_vels = []

    for intr in intruders:
        pos, vel = intr.get_state()
        state = controller.get_state(ego_pos, pos, vel)

        states.append(state)
        intruder_positions.append(pos)
        intruder_vels.append(vel)

    # Get coordinated actions
    actions = controller.select_action(states)

    # Apply actions as delta-velocities
    for intr, act in zip(intruders, actions):
        current_vel = intr.prim.get_linear_velocity()
        steering_force = act.detach().cpu().numpy() * 0.5
        new_vel = current_vel + steering_force
        speed = np.linalg.norm(new_vel)
        if speed > 6.0:
            new_vel = (new_vel / speed) * 6.0
        intr.prim.set_linear_velocity(new_vel)

    # Compute adversarial reward
    rewards = controller.compute_reward(ego_pos, intruder_positions)

    # Train
    loss = controller.train_step(states, actions, rewards)

    return loss