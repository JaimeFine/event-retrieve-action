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
