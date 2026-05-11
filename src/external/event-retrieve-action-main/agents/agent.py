from .stabilizer import LyapunovStabilizer
from .encoder import EventEncoder
from .bank import KnowledgeBank

import torch.nn as nn
import torch
from macro import device

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
        with torch.no_grad():
            # Perform SVD: Psi = U * S * V^T
            U, S, Vh = torch.linalg.svd(self.Psi, full_matrices=False)
            # Clamp the singular values (eigenvalues for symmetric matrices)
            # so they never reach or exceed 1.0
            S = S.clamp(max=margin)
            # Reconstruct the stable matrix and overwrite the parameter
            self.Psi.copy_(U @ torch.diag(S) @ Vh)

    def clustered_bayesian_selection(
        self, valid_actions, valid_weights, sim_threshold=0.8
    ):
        B = valid_actions.shape[0]
        if B == 0: return torch.zeros(3, device=device)
        if B == 1: return valid_actions[0]

        # 1. Normalize actions to get directional vectors
        dirs = valid_actions / (valid_actions.norm(dim=1, keepdim=True) + 1e-6)
        
        # O(1) step pairwise cosine similarity matrix
        sim_matrix = dirs @ dirs.t()
        # Adjacency matrix for cluster membership
        adjacency = sim_matrix > sim_threshold
        # Compute cluster weights (sum of weights per row cluster)
        cluster_weights = adjacency.float() @ valid_weights
        # Select best cluster index
        best_idx = torch.argmax(cluster_weights)
        win_mask = adjacency[best_idx]

        win_actions = valid_actions[win_mask]
        win_weights = valid_weights[win_mask]

        win_weights = win_weights / (win_weights.sum() + 1e-6)
        final_action = (win_weights.unsqueeze(1) * win_actions).sum(dim=0)

        return final_action
    
    def select_action(self, event_data, k=5):
        z_t = self.encoder(event_data)

        if self.memory.latents.shape[0] == 0:
            return torch.zeros(3, device=device), z_t, None, None, None
        
        weights, actions, _, index = self.memory.retrieve(z_t, k=k)

        if weights is None:
            # fallback action
            return torch.zeros(3, device=device), z_t, None, None, None

        # [GPU OPTIMIZATION] Stack actions to compute transitions
        # in one parallel forward pass
        if actions.dim() == 1: actions = actions.unsqueeze(0)

        # Broadcasted matrix multiplication
        z_next_pred_batch = z_t @ self.Psi + actions @ self.Gamma.t()

        # Vectorized Stability Check
        stable_mask = self.stabilizer.is_stable_batch(z_t, z_next_pred_batch)

        # Filter tensors using the boolean mask
        valid_actions = actions[stable_mask]
        valid_weights = weights[stable_mask]

        # --- Clustered Bayesian Selection ---
        if valid_actions.shape[0] == 0:
            final_action = actions[0]
        else:
            final_action = self.clustered_bayesian_selection(
                valid_actions, valid_weights
            )

        return final_action, z_t, valid_actions, valid_weights, index
    