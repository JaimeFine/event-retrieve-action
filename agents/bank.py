import torch
from macro import device

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
            self._cached_matrix = torch.stack(
                self.latent_codes
            ).squeeze(1).to(device)

            # print(f"Memory Bank synced: {self._cached_matrix.shape[0]} samples.")

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