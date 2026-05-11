# Isaac Sim Headless Template: Intruder & Environment

# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os
from omni.isaac.kit import SimulationApp
from torch.utils.data import DataLoader

# 1. Start the simulation headless
simulation_app = SimulationApp({"headless": True})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from omni.isaac.core import World   # NOTE: World is the library
from omni.isaac.core.objects import DynamicSphere

# NOTE: Are there any libraries or tool that can directly simulate the behavior
# of the natural intruders, or we don't have to? Because we are checking the
# relative robustness, where baseline and the framework are under the same
# conditions?
"""
For the Comparative Evaluation, we need strict scientific control. If we use a
randomized "natural" behavior library, we can't guarantee that PPO, acados, and
our Retrieval framework are facing the exact same obstacle trajectory. Hardcoding
deterministic kinematic paths (e.g., a parameterized sine wave or a linear
intercept vector) using numpy is the correct, rigorous approach for benchmarking.
"""

# ==================================
# 1. The Event-Centric Architecture
# ==================================

class EventEncoder(nn.Module):
    def __init__(self, input_dim=13, latent_dim=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, event_list):
        # Check if we are processing a Batch (3D) or Single Item (2D)
        is_batched = event_list.dim() == 3
        
        x = self.phi(event_list)
        
        if is_batched:
            # Batched processing: x is (Batch, N, 256)
            weights = 1.0 / (torch.norm(x, dim=2, keepdim=True) + 1e-6)
            # Sum across the sequence length (dim=1) for permutation invariance
            x = torch.sum(x * weights, dim=1) 
        else:
            # Single item processing: x is (N, 256)
            weights = 1.0 / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
            x = torch.sum(x * weights, dim=0, keepdim=True) 
            
        z_t = self.rho(x)
        return z_t
    
class KnowledgeBank:
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim
        self.maneuvers = []
        self.latent_codes = []
        self.rewards = []
        self.next_latents = []

        # Cached tensor for fast searching
        self._cached_matrix = None

    def add_experiences(self, z_i, a_i, r_i=None, z_next=None):
        self.maneuvers.append(a_i.detach())
        self.latent_codes.append(z_i.detach())
        if r_i is not None:
            self.rewards.append(r_i.detach())
        if z_next is not None:
            self.next_latents.append(z_next.detach())

        # Invalidate cache since data changed
        self._cached_matrix = None
        
    def build_index(self):
        if len(self.latent_codes) > 0:
            # [GPU OPTIMIZATION] Stack and cache directly on device
            self._cached_matrix = torch.stack(self.latent_codes).squeeze(1).to(device)

            print(f"Memory Bank synced: {self._cached_matrix.shape[0]} samples.")

    def retrieve(self, z_t, k=5, tau=0.1):
        if self._cached_matrix is None:
            self.build_index()

        actual_k = min(k, self._cached_matrix.shape[0])
        z_query = z_t.detach().squeeze(0).to(device)

        distances = torch.cdist(
            z_query.view(1, -1), self._cached_matrix, p=2
        ).squeeze(0)
        topk_values, topk_indices = torch.topk(
            distances, actual_k, largest=False
        )

        retrieved_actions = [self.maneuvers[i] for i in topk_indices]
        retrieved_latents = [self.latent_codes[i] for i in topk_indices]

        weights = 1.0 / (topk_values + 1e-8)
        weights /= weights.sum()
        
        return weights, retrieved_actions, retrieved_latents
    
class LyapunovStabilizer(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.P = nn.Parameter(torch.eye(latent_dim), requires_grad=False)

    def get_energy(self, z):
        return torch.mm(torch.mm(z, self.P), z.t())
    
    def is_stable(self, z_current, z_next_pred_batch):
        """
        [GPU OPTIMIZATION] Vectorized Lyapunov stacility check.
        Computes energy states for all retrieved maneuvers simultaneously.
        """
        v_curr = self.get_energy(z_current)

        # Efficient batched energy calculation
        P_z_next = torch.mm(z_next_pred_batch, self.P)
        v_next_batch = torch.sum(P_z_next * z_next_pred_batch, dim=1).view(-1, 1)

        stable_mask = (v_next_batch < v_curr).squeeze(1)

        return stable_mask

class EventCentricAgent(nn.Module):
    def __init__(self, latent_dim=128, action_dim=3):
        super().__init__()
        self.encoder = EventEncoder(latent_dim=latent_dim)
        self.memory = KnowledgeBank(latent_dim=latent_dim)
        self.stabilizer = LyapunovStabilizer(latent_dim=latent_dim)

        self.Psi = nn.Parameter(torch.eye(latent_dim) * 0.5)
        self.Gamma = nn.Parameter(torch.zeros(latent_dim, action_dim))
        self.to(device)

    def enforce_contractive_dynamics(self, margin=0.99):
        """
        Enforces rho(Psi) < 1 for Lyapunov stability.
        Projects the Psi matrix back into the stable region using SVD.
        Call this after optimizer.step() during the training loop.
        """
        with torch.no_grad():
            # Perform SVD: Psi = U * S * V^T
            U, S, Vh = torch.linalg.svd(self.Psi)

            # Clamp the singular values (eigenvalues for symmetric matrices)
            # so they never reach or exceed 1.0
            S_clamped = torch.clamp(S, max=margin)

            # Reconstruct the stable matrix and overwrite the parameter
            stable_Psi = torch.mm(U, torch.mm(torch.diag(S_clamped), Vh))
            self.Psi.copy_(stable_Psi)

    def clustered_bayesian_selection(
        self, valid_actions, valid_weights, sim_threshold=0.8
    ):
        """
        Prevents "average-to-collision" by clustering
        directional intent using cosine similarity.
        """
        """
        [GPU OPTIMIZATION] Cosine similarity clustering
        using parallel matrix operations.
        """
        B = valid_actions.shape[0]
        if B == 0: return torch.zeros(3, device=device)
        if B == 1: return valid_actions[0]

        # 1. Normalize actions to get directional vectors
        norms = torch.norm(valid_actions, dim=1, keepdim=True)
        dirs = valid_actions / (norms + 1e-6)
        
        # O(1) step pairwise cosine similarity matrix
        sim_matrix = torch.mm(dirs, dirs.t())

        clusters = []   # Store dicts: {'inidices': [], 'weight_sum': tensor}
        placed = torch.zeros(B, dtype=torch.bool, device=device)

        # 2. Group into clusters based on cosine similarity
        for i in range(B):
            if placed[i]: continue
            cluster_mask = sim_matrix[i] > sim_threshold
            placed |= cluster_mask

            cluster_weights = valid_weights[cluster_mask]
            clusters.append({
                'mask': cluster_mask,
                'weight_sum': cluster_weights.sum()
            })

        # 3. Bayesian Estimation: Select the cluster with highest aggregate
        # weight W_c
        winning_cluster = max(clusters, key=lambda x: x['weight_sum'])
        win_mask = winning_cluster['mask']

        # 4. Weighted averaging ONLY within the winning cluster
        win_weights = valid_weights[win_mask]
        win_actions = valid_actions[win_mask]

        # Re-normalize weights for the winning cluster
        win_weights = (win_weights / win_weights.sum()).unsqueeze(1)

        # Final action a_t
        final_action = torch.sum(win_weights * win_actions, dim=0)
        return final_action
    
    def select_action(self, event_data, k=5):
        z_t = self.encoder(event_data)

        if len(self.memory.maneuvers) == 0:
            return torch.zeros(3, device=device), z_t, None, None
        
        weights, actions, latents = self.memory.retrieve(z_t, k=k)

        # [GPU OPTIMIZATION] Stack actions to compute transitions
        # in one parallel forward pass
        actions_mat = torch.stack(actions)  # Shape: (k, 3)
        if actions_mat.dim() == 1: actions_mat = actions_mat.unsqueeze(0)

        # Broadcasted matrix multiplication
        z_next_pred_batch = torch.mm(z_t, self.Psi) + \
            torch.mm(actions_mat, self.Gamma.t())

        # Vectorized Stability Check
        stable_mask = self.stabilizer.is_stable_batch(z_t, z_next_pred_batch)

        # Filter tensors using the boolean mask
        valid_actions = actions_mat[stable_mask]
        valid_weights = weights[stable_mask]

        # --- Clustered Bayesian Selection ---
        if valid_actions.shape[0] == 0:
            final_action = actions_mat[0]
        else:
            final_action = self.clustered_bayesian_selection(
                valid_actions, valid_weights
            )

        return final_action, z_t, valid_actions.unbind(0), valid_weights.unbind(0)
    
# =========================
# 2. Isaac Sim Environment
# =========================

class BaseIntruder:
    def __init__(self, name, position, color, radius=0.3):
        self.name = name
        self.initial_position = np.array(position)
        self.radius = radius

        self.prim = DynamicSphere(
            prim_path=f"/World/{name}",
            name=name,
            position=self.initial_position,
            radius=self.radius,
            color=np.array(color)
        )

    def apply_behavior(self):
        """To be overridden by child classes"""
        pass

    def set_state(self, position, velocity):
        """Teleports the intruder and applies a new velocity vector."""
        self.prim.set_world_pose(position=position)
        self.prim.set_linear_velocity(velocity)

    def get_state(self):
        pos, _ = self.prim.get_world_pose()
        vel = self.prim.get_linear_velocity()

        return pos, vel
    
    def reset(self):
        """
        For seeded scenarios: Zeroes out physics forces and returns the
        intruder to its exact spawn point.
        """
        self.prim.set_world_pose(position=self.initial_position)
        self.prim.set_linear_velocity(np.zeros(3))
        self.prim.set_angular_velocity(np.zeros(3))
    
class DroneIntruder(BaseIntruder):  # BLUE
    def __init__(self, name, position, velocity=None):
        super().__init__(name, position, color=np.array([1.0, 0.0, 0.0]))
        self.velocity = velocity if velocity is not None \
            else np.array([-3.0, 0.0, 0.0])

    def apply_behavior(self):
        self.prim.set_linear_velocity(self.velocity)

class BirdIntruder(BaseIntruder):   # GREEN
    def __init__(
        self, name, position, base_velocity=None,
        frequency=2.0, amplitude=2.0
    ):
        super().__init__(
            name, position, color=np.array([0.0, 1.0, 0.0]), radius=0.15
        )
        self.base_velocity = base_velocity if base_velocity is not None \
            else np.array([-2.0, 0.0, 0.0])
        self.frequency = frequency
        self.amplitude = amplitude

        self.step_count = 0
        self.dt = 0.05

    def apply_behavior(self):
        # Erratic, sinusoidal flight pattern
        self.step_count += 1
        
        t = self.step_count * self.dt
        y_vel = np.sin(t * self.frequency) * self.amplitude

        current_vel = np.copy(self.base_velocity)
        current_vel[1] = y_vel

        self.prim.set_linear_velocity(current_vel)

    def reset(self):
        super().reset()

        self.step_count = 0

class StaticObstacle(BaseIntruder):
    def __init__(self, name, position):
        super().__init__(name, position, color=np.array([0.5, 0.5, 0.5]))

    def apply_behavior(self):
        # Environment constraints (buildings, poles) do not move
        self.prim.set_linear_velocity(np.zeros(3))

class Environment:
    def __init__(self):
        self.dt = 0.05
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt)
        self.agent = EventCentricAgent(latent_dim=128)

        # Formalized Navigation Task
        self.ego_start = np.array([0.0, 0.0, 1.5])
        self.ego_goal = np.array([50.0, 20.0, 5.0])

        # Simulation State
        self.d_threshold = 5.0
        self.active_scenario_intruders = [] # Initialized for the sensors
        self.experience_buffer = []
        # For the training
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-3)

    def setup_environment(self):
        self.world.scene.add_default_ground_plane()

        # Spawn Ego Drone
        self.ego = DynamicSphere(
            prim_path="/World/ego_drone",
            name="ego_drone",
            position=np.array([0.0, 0.0, 1.5]),
            radius=0.25,
            color=np.array([0.0, 0.0, 1.0]) # Blue
        )
        self.world.scene.add(self.ego)

        # PRE-SPAWN a pool of intruders to avoid physics crashes later
        self.intruders = []
        # We spawn a mix of types in a "hidden" location far underground
        hidden_pos = [0.0, 0.0, -100.0]

        for i in range(5):
            self.intruders.append(DroneIntruder(f"pool_drone_{i}", hidden_pos))
            self.intruders.append(BirdIntruder(f"pool_bird_{i}", hidden_pos))
            self.intruders.append(StaticObstacle(f"pool_static_{i}", hidden_pos))

        for intruder in self.intruders:
            self.world.scene.add(intruder.prim)

    def load_scenario(self, seed):
        self.rng = np.random.RandomState(seed)
        self.active_scenario_intruders = []

        # Teleport everyone underground
        for intruder in self.intruders:
            intruder.set_state(np.array([0.0, 0.0, -100.0]), np.zeros(3))

        self.ego.set_world_pose(position=self.ego_start)
        self.ego.set_linear_velocity(np.zeros(3))

    def detection(self):
        """Creates the Event List E_t from teh simulation state."""
        ego_pos, _ = self.ego.get_world_pose()
        ego_vel = self.ego.get_linear_velocity()
        event_list = []

        for intruder in self.active_scenario_intruders:
            pos, vel = intruder.get_state()
            dist = np.linalg.norm(pos - ego_pos)

            if dist <= self.d_threshold:
                # Use RELATIVE position and RELATIVE velocity
                rel_pos = pos - ego_pos
                rel_vel = vel - ego_vel

                if isinstance(intruder, DroneIntruder):
                    type_id = 0.0
                elif isinstance(intruder, BirdIntruder):
                    type_id = 1.0
                else:
                    type_id = 2.0

                # Concatenate intruder event with S_global (ego state)
                # Vector: [
                # id, rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz,
                # ego_vx, ego_vy, ego_vz, goal_rel_x, goal_rel_y, goal_rel_z
                # ]
                rel_goal = (self.ego_goal - ego_pos)
                s_global = ego_vel.tolist() + rel_goal.tolist()
                event = [type_id] + rel_pos.tolist() + \
                    rel_vel.tolist() + s_global
                event_list.append(event)

        return torch.tensor(
            event_list, dtype=torch.float32, device=device
        ) if event_list else None

    def architecture(self, event_list):
        """Passes E_t to the Agent to get Action a_t"""
        action, z_t, _, _ = self.agent.select_action(event_list, k=5)

        # Apply the computed maneuver to the ego drone
        act_np = action.detach().cpu().numpy()
        self.ego.set_linear_velocity(act_np)

        print(f"[ACTION COMPUTED] Applied Action: {act_np}")

    def train_agent(self, batch_size=16, epochs=5):
        if len(self.experience_buffer) < batch_size:
            return
        
        for _ in range(epochs):
            # [GPU OPTIMIZATION] Vectorized experience sampling
            indices = np.random.choice(
                len(self.experience_buffer), batch_size, replace=False
            )
            batch = [self.experience_buffer[i] for i in indices]

            self.optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            valid_samples = 0

            for (E_t, a_t, r_t, E_next) in batch:
                if E_t is None or E_next is None: continue
                z_t = self.agent.encoder(E_t).detach()

                # --- Retrieve ---
                weights, actions, latents = self.agent.memory.retrieve(z_t, k=5)
                if len(actions) == 0: continue

                # --- Physics consistency ---
                # Vectorized physics consistency penalty
                weights_t = torch.stack(weights)
                latents_t = torch.stack(latents)
                R_phys = torch.sum(weights_t * torch.norm(z_t - latents_t, dim=1))

                # --- Performance ---
                log_probs = torch.log(torch.stack(weights) + 1e-8)
                J_perf = torch.sum(log_probs) * r_t

                # Setting the heuristic lambda
                lambda_phys = 1.0
                lambda_perf = 1.0

                # Total loss
                batch_loss = batch_loss + lambda_phys * R_phys - \
                    lambda_perf * J_perf
                valid_samples += 1

                if valid_samples > 0:
                    (batch_loss / valid_samples).backward()
                    self.optimizer.step()
                    self.agent.enforce_contractive_dynamics()

            total_loss = torch.stack(batch_loss).mean()

            # --- Optimize ---
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Stability
            self.agent.enforce_contractive_dynamics()

        print("[TRAINING] Loss:", total_loss.item())

    def run(self, steps=200):   # NOTE: 200 for 0.5 dt is 10 seconds
        self.world.reset()
        self.experience_buffer = [] # Stores S_t

        for i in range(steps):
            for intruder in self.intruders:
                intruder.apply_behavior()

            # 1. Perception: Get the state at time t
            event_list = self.detection()

            if event_list is None:
                # No threats
                self.world.step(render=False)
                continue

            # 2. Decision Making: Calculate Action at time t
            action, z_t, _, _ = self.agent.select_action(event_list, k=5)
            act_np = action.detach().cpu().numpy()
            self.ego.set_linear_velocity(act_np)

            # 3. Physics: Step the environment to apply a_t
            self.world.step(render=False)

            # 4. Perception: Get the resulting state at time t+1
            event_next = self.detection()

            # 5. Reward: Calculate r_t (NOTE: A simple heuristic for now)
            if event_next is None:
                reward = 1.0    # Successfully cleared the threat
            else:
                reward = -0.1   # Small penalty for still being near a threat

            # 6. Logging: Save the Status Code
            self.experience_buffer.append(
                (event_list, action, reward, event_next)
            )
  
            # NOTE: Careful with here:
            if i % 20 == 0:
                self.train_agent()

            z_next = self.agent.encoder(
                event_next
            ) if event_next is not None else None
            self.agent.memory.add_experiences(
                z_t.detach(),
                action.detach(),
                reward,
                z_next.detach() if z_next is not None else None
            )

            print(f"[LOGGED S_t] Reward: {reward}")

        print("[COMPLETE] Collected {len(self.experience_buffer)} experiences.")
        simulation_app.close()

# ======================
# Pretraining functions
# ======================

def collate_events(batch):
    """Pads variable length event lists into a uniform batched tensor."""
    E_list = [item[0] for item in batch]
    a_list = [item[1] for item in batch]
    
    # Pad E_list to shape (Batch, Max_N_in_batch, 13)
    E_padded = pad_sequence(E_list, batch_first=True, padding_value=0.0)
    a_tensor = torch.stack(a_list)
    
    return E_padded, a_tensor

def generate_synthetic_next_state(E, a, dt=0.05):
    """
    Approximates E_next by applying action 'a' to current state 'E'.
    E structure: [type, rel[3], rel_v[3], ego_v[3], ...]
    """
    E_next = E.clone()

    # 1. Update relative positions: pos_next = pos - (v_ego_cmd - v_intruder_rel) * dt
    # This is a simplification but keeps the physics consistency
    E_next[:, 1:4] = E[:, 1:4] - (a - E[:, 4:7]) * dt
    
    # 2. Update ego velocity to the commanded action
    E_next[:, 7:10] = a

    return E_next

def pretrain_encoder(agent, dataset, batch_size=64, epochs=10):
    optimizer = torch.optim.Adam(agent.encoder.parameters(), lr=1e-3)
    
    print("Building initial memory...")
    build_memory(agent, dataset)

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_events
    )

    for epoch in range(epochs):
        agent.train()
        total_loss = 0

        for E_batch, a_batch in loader:
            E_batch = E_batch.to(device)      # (B, N, 13)
            a_batch = a_batch.to(device)      # (B, action_dim)

            optimizer.zero_grad()

            # =========================
            # 1. Encoder Forward (GPU parallel)
            # =========================
            z_batch = agent.encoder(E_batch)  # (B, latent_dim)

            # =========================
            # 2. Metric Loss (vectorized)
            # =========================
            z_shifted = torch.roll(z_batch, shifts=1, dims=0)
            E_shifted = torch.roll(E_batch, shifts=1, dims=0)

            E_mean = E_batch.mean(dim=1)
            E_shifted_mean = E_shifted.mean(dim=1)

            d_latent = torch.norm(z_batch - z_shifted, dim=1)
            d_phys = torch.norm(E_mean - E_shifted_mean, dim=1)

            loss_metric = torch.abs(d_latent - d_phys).mean()

            # =========================
            # 3. Imitation Loss (FIXED)
            # =========================
            # IMPORTANT: do NOT use select_action here
            # Instead learn a simple mapping head

            if not hasattr(agent, "pretrain_head"):
                agent.pretrain_head = nn.Linear(z_batch.shape[1], a_batch.shape[1]).to(device)

            a_pred = agent.pretrain_head(z_batch)

            loss_imitation = F.mse_loss(a_pred, a_batch)

            # =========================
            # 4. Total Loss
            # =========================
            loss = 0.5 * loss_metric + 1.0 * loss_imitation

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * E_batch.size(0)

        print(f"[PRETRAIN] Epoch {epoch} | Loss: {total_loss / len(dataset):.4f}")

        if epoch < epochs - 1:
            build_memory(agent, dataset)

def build_memory(agent, dataset):
    """
    [GPU OPTIMIZATION] Batches the memory initialization to
    prevent single-tensor sequential blocking.
    """
    dt = 0.05
    agent.eval()    # Eval mode for memory building
    # Clear old memory
    maneuvers, latent_codes, next_latents = [], [], []

    with torch.no_grad():
        for E, a in dataset:
            E = E.to(device)
            a = a.to(device)

            z = agent.encoder(E)
            E_next = generate_synthetic_next_state(E, a, dt)
            z_next = agent.encoder(E_next)

            latent_codes.append(z)
            maneuvers.append(a)
            next_latents.append(z_next)

    # Injecting directly into the knowledge bank to avoid internal overhead
    agent.memory.maneuvers = maneuvers
    agent.memory.latent_codes = latent_codes
    agent.memory.next_latents = next_latents
    agent.memory.rewards = [torch.tensor([1.0], device=device)] * len(dataset)
    
    agent.memory.build_index()
    print("Knowledge Bank M initialized and indexed.")

def save_model(agent, path="agent_pretrained.pt"):
    torch.save({
        "encoder": agent.encoder.state_dict(),
        "Psi": agent.Psi.data,
        "Gamma": agent.Gamma.data,
    }, path)

    print(f"[SAVE] Model saved to {path}")

if __name__ == "__main__":
    sim = Environment()

    if os.path.exists("expert_dataset.pt"):
        print("Loading expert data...")
        dataset = torch.load("expert_dataset.pt")

        print("Initializing Memory Bank...")
        build_memory(sim.agent, dataset)

        print("Starting Pretraining...")
        pretrain_encoder(sim.agent, dataset)

        save_model(sim.agent)

        print("Re-building Memory")
        build_memory(sim.agent, dataset)
    else:
        print("Error: expert_dataset.pt not found. Run expert.py first!")

"""
OUTPUT:
Loading expert data...
Initializing Memory Bank...
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
Starting Pretraining...
Building initial memory...
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 0 | Loss: 1.9713
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 1 | Loss: 1.1241
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 2 | Loss: 0.9924
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 3 | Loss: 0.9211
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 4 | Loss: 0.8347
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 5 | Loss: 0.7835
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 6 | Loss: 0.6940
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 7 | Loss: 0.6270
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 8 | Loss: 0.5623
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
[PRETRAIN] Epoch 9 | Loss: 0.5253
[SAVE] Model saved to agent_pretrained.pt
Re-building Memory
Memory Bank synced: 27075 samples.
Knowledge Bank M initialized and indexed.
"""