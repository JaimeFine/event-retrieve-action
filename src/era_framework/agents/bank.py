import torch
from macro import device

class KnowledgeBank:
    def __init__(self, latent_dim=128, action_dim=3, capacity=50000):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.capacity = capacity    # NOTE: Parameter!!!

        # GPU tensors
        self.latents = torch.empty((0, latent_dim), device=device)
        self.actions = torch.empty((0, action_dim), device=device)
        self.reliability = torch.empty((0, 1), device=device)

    def add_experiences(self, z_i, a_i, reward=None, z_next=None, rel=1.0):
        z_i = z_i.detach().to(device).view(1, -1)
        a_i = a_i.detach().float().to(device).view(1, -1)
        rel = torch.tensor([[rel]], device=device)

        if rel is None:
            rel = torch.clamp(torch.tensor([[reward]]), 0.0, 1.0)
        else:
            rel = torch.tensor([[rel]], device=device)

        self.latents = torch.cat([self.latents, z_i], dim=0)
        self.actions = torch.cat([self.actions, a_i], dim=0)
        self.reliability = torch.cat([self.reliability, rel], dim=0)

        if self.latents.shape[0] > self.capacity:
            self.latents = self.latents[1:]
            self.actions = self.actions[1:]
            self.reliability = self.reliability[1:]

    def penalize_by_indices(self, indices, factor):
        if indices is not None and len(indices) > 0:
            self.reliability[indices] *= factor

    def retrieve(self, z_t, k=5, tau=0.1):
        if self.latents.shape[0] == 0:
            return None, None, None, None
        
        z_query = z_t.detach().to(device).view(1, -1)

        z_norm = (z_query ** 2).sum(dim=1, keepdim=True)
        mem_norm = (self.latents ** 2).sum(dim=1)
        distances = z_norm + mem_norm - 2 * (z_query @ self.latents.t())
        distances = distances.squeeze(0)

        actual_k = min(k, self.latents.shape[0])
        topk_values, topk_indices = torch.topk(
            distances, actual_k, largest=False
        )

        retrieved_actions = self.actions[topk_indices]
        retrieved_latents = self.latents[topk_indices]

        rel_scores = self.reliability[topk_indices].squeeze(-1)
        logits = (-topk_values / tau) + torch.log(rel_scores + 1e-8)
        weights = torch.softmax(logits, dim=0)

        return weights, retrieved_actions, retrieved_latents, topk_indices
